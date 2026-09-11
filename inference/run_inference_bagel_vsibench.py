#!/usr/bin/env python3
"""
Run inference on VSI-Bench using BAGEL model.
Extracts 8 uniform frames from each video and passes them as multi-image input.
Supports checkpointing to resume from interruptions.
"""

import argparse
import json
import os
import re
import random
import threading
import zipfile
from typing import Optional, List, Dict, Any
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from PIL import Image
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

# --- ensure repo root is importable (this driver lives in inference/) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end bootstrap ---
from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from data.video_utils import sample_mp4_frames
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer


VISUAL_ONLY_THINK_SYSTEM_PROMPT = (
    "Think visually to answer the question. "
    "Enclose your visual thinking within <image_start> </image_end>."
)

INTERLEAVED_THINK_SYSTEM_PROMPT = (
    "Let's think step by step to answer the question. "
    "For text-based thinking, enclose the process within <think> </think>, "
    "e.g. <think> thinking process here </think>. "
    "For visual thinking, enclose the content within <image_start> </image_end>, "
    "e.g. <image_start> thinking image here </image_end>. "
    "Finally conclude with the final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)

TEXT_ONLY_THINK_SYSTEM_PROMPT = (
    "Let's think step by step to answer the question. "
    "Enclose your thinking process within <think> </think> tags. "
    "Finally conclude with the final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)

NO_THINKING_SYSTEM_PROMPT = (
    "Answer the question directly. "
    "Provide your final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)

THINKING_MODE_PROMPTS = {
    "visual_only_thinking": VISUAL_ONLY_THINK_SYSTEM_PROMPT,
    "interleaved_thinking": INTERLEAVED_THINK_SYSTEM_PROMPT,
    "text_only_thinking": TEXT_ONLY_THINK_SYSTEM_PROMPT,
    "no_thinking": NO_THINKING_SYSTEM_PROMPT,
}

# Question types by answer format
LETTER_ANSWER_TYPES = {
    "object_rel_distance",
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "obj_appearance_order",
    "route_planning",
}
NUMERIC_ANSWER_TYPES = {
    "object_counting",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
}


def extract_answer(model_output):
    """Extract A/B/C/D letter from model output."""
    if model_output is None:
        return None

    text = model_output.replace("**", "").strip()

    answer_tag_match = re.search(r"<answer>\s*([A-D])\s*</answer>", text, re.IGNORECASE)
    if answer_tag_match:
        return answer_tag_match.group(1).upper()

    strict_match = re.search(r"(?:Final\s+Answer|Answer):\s*([A-D])", text, re.IGNORECASE)
    if strict_match:
        return strict_match.group(1).upper()

    bracket_match = re.search(r"[\(\[]([A-D])[\)\]]", text)
    if bracket_match:
        return bracket_match.group(1).upper()

    boxed_match = re.search(r"\\boxed\{([A-D])\}", text)
    if boxed_match:
        return boxed_match.group(1).upper()

    last_letter_match = re.findall(r"\b([A-D])\b", text)
    if last_letter_match:
        return last_letter_match[-1].upper()

    return None


def extract_number(model_output):
    """Extract a numeric value from model output."""
    if model_output is None:
        return None

    text = model_output.replace("**", "").strip()

    # Try <answer> tag first
    answer_tag = re.search(r"<answer>\s*([\d.]+)\s*</answer>", text, re.IGNORECASE)
    if answer_tag:
        try:
            return float(answer_tag.group(1))
        except ValueError:
            pass

    # Try "Answer: X" pattern
    answer_match = re.search(r"(?:Final\s+Answer|Answer):\s*([\d.]+)", text, re.IGNORECASE)
    if answer_match:
        try:
            return float(answer_match.group(1))
        except ValueError:
            pass

    # Last number in text, but skip numbers that appear in "[Round N]" prefixes
    clean_text = re.sub(r"\[Round \d+\]", "", text)
    numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", clean_text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass

    return None


def compute_mra(predicted: Optional[float], gt: float) -> float:
    """Mean Relative Accuracy averaged over thresholds {0.5, 0.55, ..., 0.95}."""
    if predicted is None or gt == 0:
        return 0.0
    rel_error = abs(predicted - gt) / abs(gt)
    thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5, 0.55, ..., 0.95
    return sum(1.0 if rel_error < (1.0 - theta) else 0.0 for theta in thresholds) / 10.0


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CheckpointManager:
    def __init__(self, checkpoint_path: str, save_interval: int = 10):
        self.checkpoint_path = checkpoint_path
        self.save_interval = save_interval
        self.lock = threading.Lock()
        self.results = {}
        self.completed_count = 0

    def load_checkpoint(self) -> Dict[str, Dict]:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r") as f:
                    data = json.load(f)
                    all_results = data.get("results", [])
                    valid_results = [r for r in all_results if r.get("final_answer_text", "").strip()]
                    empty_count = len(all_results) - len(valid_results)
                    self.results = {r["sample_id"]: r for r in valid_results}
                    print(f"Loaded checkpoint with {len(self.results)} completed samples")
                    if empty_count > 0:
                        print(f"Filtered out {empty_count} samples with empty responses")
                    return self.results
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load checkpoint: {e}")
        return {}

    def get_completed_sample_ids(self) -> set:
        return set(self.results.keys())

    def add_result(self, result: Dict):
        with self.lock:
            self.results[result["sample_id"]] = result
            self.completed_count += 1
            if self.completed_count % self.save_interval == 0:
                self._save_checkpoint()

    def _save_checkpoint(self):
        temp_path = self.checkpoint_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump({"results": list(self.results.values()), "total_completed": len(self.results)}, f)
        os.replace(temp_path, self.checkpoint_path)

    def save_final(self):
        with self.lock:
            self._save_checkpoint()
            print(f"Checkpoint saved: {len(self.results)} samples")

    def get_all_results(self) -> List[Dict]:
        with self.lock:
            return list(self.results.values())


def load_model(model_path: str, max_mem_per_gpu: str = "80GiB", vit_min_size: int = 512):
    print(f"Loading model from {model_path}...")

    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, vit_min_size, 14)

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    max_memory = {i: max_mem_per_gpu for i in range(num_gpus)}

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        'language_model.model.embed_tokens', 'time_embedder', 'latent_pos_embed',
        'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed'
    ]
    if num_gpus == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device if k in device_map else "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    checkpoint_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(model_path, "ema.safetensors")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = model_path

    model = load_checkpoint_and_dispatch(
        model, checkpoint=checkpoint_path, device_map=device_map,
        offload_buffers=True, dtype=torch.bfloat16, force_hooks=True,
        offload_folder="/tmp/offload",
    )
    model = model.eval()
    print('Model loaded successfully')
    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


class BAGELVSIBenchInference:
    def __init__(
        self,
        model_path: str,
        max_mem_per_gpu: str = "80GiB",
        vit_min_size: int = 512,
        max_think_token_n: int = 4096,
        do_sample: bool = True,
        text_temperature: float = 0.3,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 2.0,
        cfg_interval: tuple = (0.0, 1.0),
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "text_channel",
        image_shapes: tuple = None,
    ):
        model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(
            model_path, max_mem_per_gpu, vit_min_size
        )
        self.inferencer = InterleaveInferencer(
            model=model, vae_model=vae_model, tokenizer=tokenizer,
            vae_transform=vae_transform, vit_transform=vit_transform,
            new_token_ids=new_token_ids
        )
        self.inference_hyper = {
            'max_think_token_n': max_think_token_n,
            'do_sample': do_sample,
            'text_temperature': text_temperature,
            'cfg_text_scale': cfg_text_scale,
            'cfg_img_scale': cfg_img_scale,
            'cfg_interval': list(cfg_interval),
            'timestep_shift': timestep_shift,
            'num_timesteps': num_timesteps,
            'cfg_renorm_min': cfg_renorm_min,
            'cfg_renorm_type': cfg_renorm_type,
            'image_shapes': image_shapes,
        }

    def run_single_sample(
        self,
        frames: List[Image.Image],
        question: str,
        ground_truth: str,
        question_type: str,
        think: bool = True,
        thinking_mode: str = "text_only_thinking",
        sample_id: str = "",
        output_dir: str = None,
    ) -> Dict[str, Any]:
        NUM_RETRIES = 3

        system_prompt = THINKING_MODE_PROMPTS[thinking_mode]
        full_prompt = system_prompt + "\n\nQUESTION: " + question

        is_numeric = question_type in NUMERIC_ANSWER_TYPES

        for retry in range(NUM_RETRIES):
            try:
                input_list = frames + [full_prompt]
                output_list = self.inferencer(
                    input_list=input_list,
                    understanding_output=False,
                    think=think,
                    **self.inference_hyper
                )

                model_output_result = ""
                text_round = 0
                image_round = 0
                saved_image_paths = []

                for out_item in output_list:
                    if isinstance(out_item, str):
                        model_output_result += f"[Round {text_round}]\n{out_item}\n\n"
                        text_round += 1
                    elif isinstance(out_item, Image.Image):
                        if output_dir is not None:
                            os.makedirs(output_dir, exist_ok=True)
                            fname = f"{sample_id}_round_{image_round}.png" if sample_id else f"image_round_{image_round}.png"
                            image_path = os.path.join(output_dir, fname)
                            out_item.save(image_path)
                            saved_image_paths.append(image_path)
                        image_round += 1

                model_output_result = model_output_result.strip()

                if is_numeric:
                    predicted_num = extract_number(model_output_result)
                    try:
                        gt_num = float(ground_truth)
                    except ValueError:
                        gt_num = None
                    mra = compute_mra(predicted_num, gt_num) if gt_num is not None else 0.0
                    return {
                        "final_answer_text": model_output_result,
                        "predicted_answer": str(predicted_num) if predicted_num is not None else None,
                        "ground_truth": ground_truth,
                        "question_type": question_type,
                        "is_numeric": True,
                        "accuracy": None,
                        "mra": mra,
                        "saved_image_paths": saved_image_paths,
                    }
                else:
                    predicted = extract_answer(model_output_result)
                    acc = 1.0 if predicted is not None and predicted.upper() == ground_truth.upper() else 0.0
                    return {
                        "final_answer_text": model_output_result,
                        "predicted_answer": predicted,
                        "ground_truth": ground_truth,
                        "question_type": question_type,
                        "is_numeric": False,
                        "accuracy": acc,
                        "mra": None,
                        "saved_image_paths": saved_image_paths,
                    }

            except Exception as e:
                if retry < NUM_RETRIES - 1:
                    print(f"Error, retrying ({retry+1}/{NUM_RETRIES}): {e}")
                else:
                    print(f"All retries failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return {
                        "final_answer_text": "",
                        "predicted_answer": None,
                        "ground_truth": ground_truth,
                        "question_type": question_type,
                        "is_numeric": is_numeric,
                        "accuracy": 0.0 if not is_numeric else None,
                        "mra": 0.0 if is_numeric else None,
                        "error": str(e),
                    }

        return {
            "final_answer_text": "",
            "predicted_answer": None,
            "ground_truth": ground_truth,
            "question_type": question_type,
            "is_numeric": is_numeric,
            "accuracy": 0.0 if not is_numeric else None,
            "mra": 0.0 if is_numeric else None,
        }


def ensure_videos_extracted(data_dir: str):
    """Extract zip files if video directories don't exist yet."""
    for source in ["arkitscenes", "scannet", "scannetpp"]:
        video_dir = os.path.join(data_dir, source)
        zip_path = os.path.join(data_dir, f"{source}.zip")
        if not os.path.isdir(video_dir):
            if os.path.exists(zip_path):
                print(f"Extracting {zip_path}...")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(data_dir)
                print(f"Done extracting {source}")
            else:
                print(f"Warning: {zip_path} not found and {video_dir} does not exist")


def compute_metrics(results: List[Dict]) -> Dict:
    letter_results = [r for r in results if not r.get("is_numeric")]
    numeric_results = [r for r in results if r.get("is_numeric")]

    overall_accuracy = (
        sum(r["accuracy"] for r in letter_results) / len(letter_results)
        if letter_results else 0.0
    )
    overall_mra = (
        sum(r["mra"] for r in numeric_results) / len(numeric_results)
        if numeric_results else 0.0
    )

    per_type = {}
    all_types = set(r["question_type"] for r in results)
    for qt in all_types:
        qt_results = [r for r in results if r["question_type"] == qt]
        if qt in NUMERIC_ANSWER_TYPES:
            per_type[qt] = {
                "mra": sum(r["mra"] for r in qt_results) / len(qt_results),
                "count": len(qt_results),
            }
        else:
            per_type[qt] = {
                "accuracy": sum(r["accuracy"] for r in qt_results) / len(qt_results),
                "count": len(qt_results),
            }

    return {
        "overall_accuracy_letter": overall_accuracy,
        "overall_mra_numeric": overall_mra,
        "total_letter_samples": len(letter_results),
        "total_numeric_samples": len(numeric_results),
        "per_question_type": per_type,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on VSI-Bench using BAGEL model"
    )
    parser.add_argument(
        "--data_dir",
        default="/path/to/scratch/datasets/VSI-Bench",
    )
    parser.add_argument(
        "--dataset_file",
        default="test_debiased.parquet",
        help="Parquet or JSONL file relative to data_dir",
    )
    parser.add_argument(
        "--model_path",
        default="/path/to/scratch/models/BAGEL-7B-MoT",
    )
    parser.add_argument("--max_mem_per_gpu", default="80GiB")
    parser.add_argument("--vit_min_size", type=int, default=512)
    parser.add_argument(
        "--thinking_mode",
        choices=["visual_only_thinking", "interleaved_thinking", "text_only_thinking", "no_thinking"],
        default="text_only_thinking",
    )
    parser.add_argument("--think", action="store_true", default=True)
    parser.add_argument("--no_think", action="store_true")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to sample per video")
    parser.add_argument("--generated_images_dir", default=None,
                        help="Directory to save generated images. If None, images are not saved.")
    parser.add_argument(
        "--output_file",
        default="/path/to/scratch/vsibench/BAGEL/inference_results.json",
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help="Shard specification 'index/total' (e.g. '0/4')",
    )

    # Inference hyperparameters
    parser.add_argument("--max_think_token_n", type=int, default=4096)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--text_temperature", type=float, default=0.3)
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=2.0)
    parser.add_argument("--cfg_interval_start", type=float, default=0.0)
    parser.add_argument("--cfg_interval_end", type=float, default=1.0)
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument("--cfg_renorm_type", type=str, default="text_channel",
                        choices=["global", "channel", "text_channel"])
    parser.add_argument("--image_shapes", type=int, nargs=2, default=None,
                        help="(H, W) shape for generated images, e.g. 320 1024. Default: None (no generation).")
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument(
        "--extract_videos",
        action="store_true",
        help="Force extraction of zip files even if video dirs exist",
    )

    args = parser.parse_args()

    think = args.think and not args.no_think
    set_seed(args.random_seed)

    # Ensure videos are extracted
    if args.extract_videos or not os.path.isdir(os.path.join(args.data_dir, "arkitscenes")):
        ensure_videos_extracted(args.data_dir)

    # Load dataset
    dataset_path = os.path.join(args.data_dir, args.dataset_file)
    print(f"Loading dataset from {dataset_path}")
    if dataset_path.endswith(".parquet"):
        df = pd.read_parquet(dataset_path)
        all_data = df.to_dict("records")
    else:
        all_data = []
        with open(dataset_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_data.append(json.loads(line))
    print(f"Total samples: {len(all_data)}")

    if args.num_samples is not None and args.num_samples < len(all_data):
        random.seed(args.random_seed)
        all_data = random.sample(all_data, args.num_samples)
        print(f"Sampled {args.num_samples} examples")

    # Sharding
    if args.shard is not None:
        shard_idx, total_shards = map(int, args.shard.split("/"))
        all_data = sorted(all_data, key=lambda x: x.get("id", 0))
        total = len(all_data)
        size = total // total_shards
        remainder = total % total_shards
        start = shard_idx * size + min(shard_idx, remainder)
        end = start + size + (1 if shard_idx < remainder else 0)
        all_data = all_data[start:end]
        print(f"Shard {shard_idx}/{total_shards}: samples {start}-{end-1} ({len(all_data)} samples)")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    checkpoint_path = args.output_file.replace(".json", "_checkpoint.json")
    checkpoint_mgr = CheckpointManager(checkpoint_path, save_interval=args.checkpoint_interval)

    completed_sample_ids = set()
    if not args.no_resume:
        if os.path.exists(checkpoint_path):
            checkpoint_mgr.load_checkpoint()
            completed_sample_ids = checkpoint_mgr.get_completed_sample_ids()
            print(f"Resuming: {len(completed_sample_ids)} done, {len(all_data) - len(completed_sample_ids)} remaining")
        elif os.path.exists(args.output_file):
            try:
                with open(args.output_file) as f:
                    prev = json.load(f)
                for r in prev.get("results", []):
                    sid = r.get("sample_id", "")
                    if sid and r.get("final_answer_text", "").strip():
                        checkpoint_mgr.results[sid] = r
                completed_sample_ids = checkpoint_mgr.get_completed_sample_ids()
                print(f"Loaded {len(completed_sample_ids)} from output file")
            except Exception as e:
                print(f"Warning: could not load output file: {e}")

    inference_engine = BAGELVSIBenchInference(
        model_path=args.model_path,
        max_mem_per_gpu=args.max_mem_per_gpu,
        vit_min_size=args.vit_min_size,
        max_think_token_n=args.max_think_token_n,
        do_sample=args.do_sample,
        text_temperature=args.text_temperature,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=(args.cfg_interval_start, args.cfg_interval_end),
        timestep_shift=args.timestep_shift,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=args.cfg_renorm_min,
        cfg_renorm_type=args.cfg_renorm_type,
        image_shapes=tuple(args.image_shapes) if args.image_shapes else None,
    )

    def process_sample(example: Dict) -> Optional[Dict]:
        sample_id = str(example.get("id", ""))
        try:
            dataset_source = example["dataset"]
            scene_name = example["scene_name"]
            video_path = os.path.join(args.data_dir, dataset_source, f"{scene_name}.mp4")

            frames, _ = sample_mp4_frames(video_path, n_frames=args.num_frames)

            question_text = example["question"]
            options = example.get("options")
            if options is not None:
                options_str = "\n".join(str(o) for o in options)
                question_text = question_text + "\n" + options_str

            result = inference_engine.run_single_sample(
                frames=frames,
                question=question_text,
                ground_truth=str(example["ground_truth"]),
                question_type=example["question_type"],
                think=think,
                thinking_mode=args.thinking_mode,
                sample_id=sample_id,
                output_dir=args.generated_images_dir,
            )
            result["sample_id"] = sample_id
            result["question"] = question_text
            result["dataset"] = dataset_source
            result["scene_name"] = scene_name
            return result
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    samples_to_process = [ex for ex in all_data if str(ex.get("id", "")) not in completed_sample_ids]
    print(f"Samples to process: {len(samples_to_process)}")
    print(f"Think: {think}, Thinking mode: {args.thinking_mode}, Frames: {args.num_frames}")

    try:
        for example in tqdm(samples_to_process, desc="Running inference"):
            result = process_sample(example)
            if result:
                checkpoint_mgr.add_result(result)

            all_results = checkpoint_mgr.get_all_results()
            if len(all_results) > 0 and len(all_results) % 10 == 0:
                letter = [r for r in all_results if not r.get("is_numeric") and r.get("accuracy") is not None]
                numeric = [r for r in all_results if r.get("is_numeric") and r.get("mra") is not None]
                acc_str = f"acc={sum(r['accuracy'] for r in letter)/len(letter):.3f}" if letter else "acc=N/A"
                mra_str = f"mra={sum(r['mra'] for r in numeric)/len(numeric):.3f}" if numeric else "mra=N/A"
                print(f"[{len(all_results)} samples] {acc_str}, {mra_str}")

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        checkpoint_mgr.save_final()
        return

    checkpoint_mgr.save_final()
    results = checkpoint_mgr.get_all_results()
    metrics = compute_metrics(results)

    print(f"\n{'='*60}")
    print(f"VSI-Bench Inference Complete!")
    print(f"Total Samples: {len(results)}")
    print(f"Letter Accuracy: {metrics['overall_accuracy_letter']:.4f} ({metrics['total_letter_samples']} samples)")
    print(f"Numeric MRA:     {metrics['overall_mra_numeric']:.4f} ({metrics['total_numeric_samples']} samples)")
    print(f"\nPer question type:")
    for qt, m in sorted(metrics["per_question_type"].items()):
        if "accuracy" in m:
            print(f"  {qt}: accuracy={m['accuracy']:.4f} (n={m['count']})")
        else:
            print(f"  {qt}: mra={m['mra']:.4f} (n={m['count']})")
    print(f"{'='*60}")

    output_data = {
        "config": {
            "model_path": args.model_path,
            "think": think,
            "thinking_mode": args.thinking_mode,
            "num_frames": args.num_frames,
            "num_samples": len(results),
            "text_temperature": args.text_temperature,
            "shard": args.shard,
            "dataset_file": args.dataset_file,
        },
        "metrics": metrics,
        "results": results,
    }

    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output_file}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint removed: {checkpoint_path}")


if __name__ == "__main__":
    main()
