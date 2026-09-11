#!/usr/bin/env python3
"""
Run inference on Spatial Collaboration dataset using BAGEL with dynamic visual thinking.

The model self-selects the visual thinking mode by generating a mode token before
<image_start>:
  <panoramic> <image_start> ... <image_end>  → generates at 720×2048
  <BEV> <image_start> ... <image_end>        → generates at 720×720

Uses DynamicVisualInterleaveInferencer which detects the mode token and dispatches
to gen_image() at the appropriate resolution.
"""

import argparse
import json
import os
import re
import random
import threading
from typing import Optional, List, Dict, Any
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

# --- ensure repo root is importable (this driver lives in inference/) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end bootstrap ---
from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from SpatialUnderstanding.dynamic_visual_tokens import add_dynamic_visual_tokens
from SpatialUnderstanding.dynamic_visual_inferencer import (
    DynamicVisualInterleaveInferencer,
    DYNAMIC_VISUAL_THINK_SYSTEM_PROMPT,
)


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
    "dynamic_visual_thinking": DYNAMIC_VISUAL_THINK_SYSTEM_PROMPT,
    "visual_only_thinking": VISUAL_ONLY_THINK_SYSTEM_PROMPT,
    "interleaved_thinking": INTERLEAVED_THINK_SYSTEM_PROMPT,
    "text_only_thinking": TEXT_ONLY_THINK_SYSTEM_PROMPT,
    "no_thinking": NO_THINKING_SYSTEM_PROMPT,
}


def extract_answer(model_output):
    """Extract the multiple-choice letter (A–D) from model generation."""
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


def calculate_accuracy(answer: Optional[str], correct_answer_idx: int) -> float:
    if answer is None:
        return 0.0
    return 1.0 if answer == chr(65 + correct_answer_idx) else 0.0


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
                    valid_results = []
                    empty_count = 0
                    for r in all_results:
                        if r.get("final_answer_text", "").strip():
                            valid_results.append(r)
                        else:
                            empty_count += 1
                    self.results = {r["sample_id"]: r for r in valid_results}
                    print(f"Loaded checkpoint with {len(self.results)} completed samples")
                    if empty_count > 0:
                        print(f"Filtered out {empty_count} samples with empty responses (will recompute)")
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
        checkpoint_data = {"results": list(self.results.values()), "total_completed": len(self.results)}
        temp_path = self.checkpoint_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(checkpoint_data, f)
        os.replace(temp_path, self.checkpoint_path)

    def save_final(self):
        with self.lock:
            self._save_checkpoint()
            print(f"Checkpoint saved: {len(self.results)} samples")

    def get_all_results(self) -> List[Dict]:
        with self.lock:
            return list(self.results.values())


def load_model(model_path: str, max_mem_per_gpu: str = "80GiB"):
    """Load the BAGEL model, tokenizer, and register dynamic visual mode tokens."""
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
    tokenizer, new_token_ids, _ = add_dynamic_visual_tokens(tokenizer, new_token_ids)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    max_memory = {i: max_mem_per_gpu for i in range(num_gpus)}
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed',
    ]

    if torch.cuda.device_count() == 1:
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

    print(f"Loading checkpoint from: {checkpoint_path}")
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_path,
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload",
    )
    model = model.eval()
    print("Model loaded successfully")

    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


class BAGELDynamicVisualInference:
    """Run inference using BAGEL with dynamic visual mode token selection."""

    def __init__(
        self,
        model_path: str,
        max_mem_per_gpu: str = "80GiB",
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
            model_path, max_mem_per_gpu
        )

        self.inferencer = DynamicVisualInterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
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
        answerer_image: Image.Image,
        helper_image: Image.Image,
        question: str,
        options: List[str],
        correct_answer_idx: int,
        correct_answer: str,
        question_type: str,
        think: bool = True,
        thinking_mode: str = "dynamic_visual_thinking",
        sample_id: str = "",
        output_dir: str = None,
    ) -> Dict[str, Any]:
        NUM_RETRIES = 3

        options_str = "\n".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))
        full_question = f"\nQUESTION:{question}\n\nOPTIONS:{options_str}"
        system_prompt = THINKING_MODE_PROMPTS[thinking_mode]
        full_prompt = system_prompt + "\n" + full_question

        for retry in range(NUM_RETRIES):
            try:
                input_list = [answerer_image, helper_image, full_prompt]
                output_list = self.inferencer(
                    input_list=input_list,
                    understanding_output=False,
                    think=think,
                    **self.inference_hyper,
                )

                model_output_result = ""
                text_round = 0
                image_round = 0
                saved_image_paths = []

                for out_item in output_list:
                    if isinstance(out_item, str):
                        model_output_result += f"[Round {text_round}]\n{out_item}\n\n"
                        text_round += 1
                    elif isinstance(out_item, tuple):
                        # (mode_name, Image.Image) — mode-detected image
                        mode_name, img = out_item
                        if output_dir is not None:
                            os.makedirs(output_dir, exist_ok=True)
                            tag = mode_name.strip('<>')
                            fname = f"{sample_id}_round_{image_round}_{tag}.png" if sample_id else f"image_round_{image_round}_{tag}.png"
                            path = os.path.join(output_dir, fname)
                            img.save(path)
                            saved_image_paths.append(path)
                        image_round += 1
                    elif isinstance(out_item, Image.Image):
                        # Fallback image (no mode token detected)
                        if output_dir is not None:
                            os.makedirs(output_dir, exist_ok=True)
                            fname = f"{sample_id}_round_{image_round}.png" if sample_id else f"image_round_{image_round}.png"
                            path = os.path.join(output_dir, fname)
                            out_item.save(path)
                            saved_image_paths.append(path)
                        image_round += 1

                model_output_result = model_output_result.strip()
                predicted_answer = extract_answer(model_output_result)
                accuracy = calculate_accuracy(predicted_answer, correct_answer_idx)

                return {
                    "final_answer_text": model_output_result,
                    "predicted_answer": predicted_answer,
                    "correct_answer": correct_answer,
                    "correct_answer_idx": correct_answer_idx,
                    "accuracy": accuracy,
                    "saved_image_paths": saved_image_paths,
                }

            except Exception as e:
                if retry < NUM_RETRIES - 1:
                    print(f"Error in inference, retrying ({retry+1}/{NUM_RETRIES}): {e}")
                else:
                    print(f"All retries failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return {
                        "final_answer_text": "",
                        "predicted_answer": None,
                        "correct_answer": correct_answer,
                        "correct_answer_idx": correct_answer_idx,
                        "accuracy": 0.0,
                        "error": str(e),
                    }

        return {
            "final_answer_text": "",
            "predicted_answer": None,
            "correct_answer": correct_answer,
            "correct_answer_idx": correct_answer_idx,
            "accuracy": 0.0,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using BAGEL with dynamic visual mode token selection",
    )
    parser.add_argument("--data_dir", default="/path/to/scratch/spatial_collab_dataset")
    parser.add_argument("--dataset_files", nargs="+", default=["anchor_dataset_V_Final_2000.json"])
    parser.add_argument(
        "--model_path",
        default="/path/to/scratch/models/BAGEL-7B-MoT",
        help="Path to BAGEL model directory",
    )
    parser.add_argument("--max_mem_per_gpu", default="80GiB")
    parser.add_argument(
        "--thinking_mode",
        choices=list(THINKING_MODE_PROMPTS.keys()),
        default="dynamic_visual_thinking",
    )
    parser.add_argument("--think", action="store_true", default=True)
    parser.add_argument("--no_think", action="store_true")
    parser.add_argument("--output_file", default="/path/to/scratch/spatial_collab/BAGEL/inference_results_bagel_dynamic.json")
    parser.add_argument("--generated_images_dir", default=None)
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
    parser.add_argument(
        "--image_shapes", type=int, nargs=2, default=[720, 720],
        help="Fallback (H, W) when no mode token is detected. Default: 720 720 (BEV).",
    )
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--no_resume", action="store_true")

    args = parser.parse_args()

    think = args.think and not args.no_think
    thinking_mode = args.thinking_mode

    set_seed(args.random_seed)

    all_data = []
    for dataset_file in args.dataset_files:
        file_path = os.path.join(args.data_dir, dataset_file)
        print(f"Loading data from {file_path}")
        with open(file_path, "r") as f:
            data = json.load(f)
            all_data.extend(data)
            print(f"Loaded {len(data)} samples from {dataset_file}")

    print(f"Total samples loaded: {len(all_data)}")

    if args.num_samples is not None and args.num_samples < len(all_data):
        random.seed(args.random_seed)
        all_data = random.sample(all_data, args.num_samples)
        print(f"Randomly sampled {args.num_samples} samples (seed={args.random_seed})")

    if args.shard is not None:
        try:
            shard_idx, total_shards = map(int, args.shard.split("/"))
            all_data = sorted(all_data, key=lambda x: x.get("sample_id", ""))
            total_samples = len(all_data)
            shard_size = total_samples // total_shards
            remainder = total_samples % total_shards
            start_idx = shard_idx * shard_size + min(shard_idx, remainder)
            end_idx = start_idx + shard_size + (1 if shard_idx < remainder else 0)
            all_data = all_data[start_idx:end_idx]
            print(f"Shard {shard_idx}/{total_shards}: samples {start_idx}-{end_idx-1} ({len(all_data)} samples)")
        except ValueError as e:
            print(f"Error parsing shard argument '{args.shard}': {e}")
            return

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    checkpoint_path = args.output_file.replace(".json", "_checkpoint.json")
    checkpoint_mgr = CheckpointManager(checkpoint_path, save_interval=args.checkpoint_interval)

    completed_sample_ids = set()
    if not args.no_resume:
        if os.path.exists(checkpoint_path):
            print(f"Checkpoint detected: {checkpoint_path}")
            checkpoint_mgr.load_checkpoint()
            completed_sample_ids = checkpoint_mgr.get_completed_sample_ids()
            print(f"Auto-resuming: {len(completed_sample_ids)} done, {len(all_data) - len(completed_sample_ids)} remaining")
        elif os.path.exists(args.output_file):
            try:
                with open(args.output_file, "r") as f:
                    output_data = json.load(f)
                for r in output_data.get("results", []):
                    if r.get("sample_id") and r.get("final_answer_text", "").strip():
                        checkpoint_mgr.results[r["sample_id"]] = r
                completed_sample_ids = checkpoint_mgr.get_completed_sample_ids()
                print(f"Loaded {len(completed_sample_ids)} completed samples from output file")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load output file for resume: {e}")

    inference_engine = BAGELDynamicVisualInference(
        model_path=args.model_path,
        max_mem_per_gpu=args.max_mem_per_gpu,
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

    def load_images(example: Dict[str, Any]):
        user_1_path = example["user_1_image_local_path"]
        user_2_path = example["user_2_image_local_path"]

        if user_1_path.startswith("images/"):
            user_1_image_path = os.path.join(args.data_dir, user_1_path)
        else:
            user_1_image_path = user_1_path.replace(
                "/path/to/scratch", "/path/to/scratch"
            )
        if user_2_path.startswith("images/"):
            user_2_image_path = os.path.join(args.data_dir, user_2_path)
        else:
            user_2_image_path = user_2_path.replace(
                "/path/to/scratch", "/path/to/scratch"
            )

        question = example["question_both_views"].replace(
            "both your and your partner's perspectives", "both images"
        )

        if example.get("options_user_2") is not None:
            options = example["options_user_2"]
            correct_answer_idx = example["user_2_gt_answer_idx"]
            correct_answer = example["user_2_gt_answer_text"]
        else:
            options = example.get("options_user_1", [])
            correct_answer_idx = example.get("user_1_gt_answer_idx", 0)
            correct_answer = example.get("user_1_gt_answer_text", "")

        return {
            "answerer_image": Image.open(user_1_image_path).convert('RGB'),
            "helper_image": Image.open(user_2_image_path).convert('RGB'),
            "question": question,
            "options": options,
            "correct_answer_idx": correct_answer_idx,
            "correct_answer": correct_answer,
            "question_type": example.get("question_type", ""),
            "sample_id": example.get("sample_id", ""),
        }

    def process_single_sample(example: Dict[str, Any]):
        sample_id = example.get("sample_id", "")
        try:
            loaded_data = load_images(example)
            result = inference_engine.run_single_sample(
                answerer_image=loaded_data["answerer_image"],
                helper_image=loaded_data["helper_image"],
                question=loaded_data["question"],
                options=loaded_data["options"],
                correct_answer_idx=loaded_data["correct_answer_idx"],
                correct_answer=loaded_data["correct_answer"],
                question_type=loaded_data["question_type"],
                think=think,
                thinking_mode=thinking_mode,
                sample_id=loaded_data["sample_id"],
                output_dir=args.generated_images_dir,
            )
            result["sample_id"] = loaded_data["sample_id"]
            result["question_type"] = loaded_data["question_type"]
            result["question"] = loaded_data["question"]
            result["options"] = loaded_data["options"]
            return result
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    samples_to_process = [
        example for example in all_data
        if example.get("sample_id", "") not in completed_sample_ids
    ]

    print(f"Running inference...")
    print(f"Samples to process: {len(samples_to_process)}")
    print(f"Think mode: {think}, Thinking mode: {thinking_mode}")

    try:
        for example in tqdm(samples_to_process, desc="Running inference"):
            result = process_single_sample(example)
            if result:
                checkpoint_mgr.add_result(result)

            all_results = checkpoint_mgr.get_all_results()
            if len(all_results) > 0 and len(all_results) % 10 == 0:
                current_acc = sum(r["accuracy"] for r in all_results) / len(all_results)
                print(f"Current accuracy: {current_acc:.4f} ({len(all_results)} samples)")

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        checkpoint_mgr.save_final()
        return

    checkpoint_mgr.save_final()
    results = checkpoint_mgr.get_all_results()

    final_accuracy = sum(r["accuracy"] for r in results) / len(results) if results else 0.0

    print(f"\n{'='*60}")
    print(f"Inference Complete!")
    print(f"Total Samples: {len(results)}")
    print(f"Overall Accuracy: {final_accuracy:.4f}")
    print(f"{'='*60}")

    output_data = {
        "config": {
            "model_path": args.model_path,
            "think": think,
            "thinking_mode": thinking_mode,
            "num_samples": len(results),
            "text_temperature": args.text_temperature,
            "shard": args.shard,
        },
        "metrics": {
            "overall_accuracy": final_accuracy,
            "total_correct": sum(r["accuracy"] for r in results),
            "total_samples": len(results),
        },
        "results": results,
    }

    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output_file}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint file removed: {checkpoint_path}")


if __name__ == "__main__":
    main()
