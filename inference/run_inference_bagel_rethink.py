#!/usr/bin/env python3
"""
Re-inference script: tests whether feeding the pre-generated "thinking image" back as
a 3rd input (understanding mode, no new generation) achieves the same accuracy as the
original visual-only generation run.

Input per sample:
  [answerer_image, helper_image, generated_thinking_image, question+options]

Output: direct answer (understanding_output=True, no image generation).

Usage:
  python run_inference_bagel_rethink.py \
    --results_file /path/to/prior_inference_results.json \
    --data_dir /path/to/scratch/spatial_collab_dataset \
    --dataset_files spatial_dataset_V_Final_2000.json \
    --model_path /path/to/model \
    --output_file /path/to/output.json
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
from inferencer import InterleaveInferencer


NO_THINKING_SYSTEM_PROMPT = (
    "Answer the question directly. "
    "Provide your final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)

TEXT_ONLY_THINK_SYSTEM_PROMPT = (
    "Let's think step by step to answer the question. "
    "Enclose your thinking process within <think> </think> tags. "
    "Finally conclude with the final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)

UNDERSTANDING_SYSTEM_PROMPT = (
    "You are given two camera views and a panoramic overview of the scene. "
    "Answer the question directly. "
    "Provide your final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)

THINKING_MODE_PROMPTS = {
    "no_thinking": NO_THINKING_SYSTEM_PROMPT,
    "text_only_thinking": TEXT_ONLY_THINK_SYSTEM_PROMPT,
    "understanding": UNDERSTANDING_SYSTEM_PROMPT,
}


def extract_answer(model_output):
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
    expected_letter = chr(65 + correct_answer_idx)
    return 1.0 if answer == expected_letter else 0.0


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
                    self.results = {r["sample_id"]: r for r in valid_results}
                    print(f"Loaded checkpoint with {len(self.results)} completed samples")
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
        'vit_pos_embed'
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
    print('Model loaded successfully')

    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


class BAGELRethinkInference:
    """
    Re-inference with pre-generated thinking image as 3rd input.
    No new image generation — understanding_output=True throughout.
    """

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
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
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
        answerer_image: Image.Image,
        helper_image: Image.Image,
        generated_image: Image.Image,
        question: str,
        options: List[str],
        correct_answer_idx: int,
        correct_answer: str,
        question_type: str,
        thinking_mode: str = "no_thinking",
        sample_id: str = "",
        original_predicted_answer: Optional[str] = None,
        original_accuracy: float = 0.0,
        generated_image_path: str = "",
    ) -> Dict[str, Any]:
        NUM_RETRIES = 3

        options_str = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)])
        full_question = f"\nQUESTION:{question}\n\nOPTIONS:{options_str}"
        system_prompt = THINKING_MODE_PROMPTS[thinking_mode]
        full_prompt = system_prompt + "\n" + full_question

        for retry in range(NUM_RETRIES):
            try:
                # Feed 3 images + question, understanding only (no image generation)
                output_list = self.inferencer(
                    input_list=[answerer_image, helper_image, generated_image, full_prompt],
                    understanding_output=True,
                    think=False,
                    **self.inference_hyper
                )

                model_output_result = ""
                for out_item in output_list:
                    if isinstance(out_item, str):
                        model_output_result += out_item

                model_output_result = model_output_result.strip()
                predicted_answer = extract_answer(model_output_result)
                accuracy = calculate_accuracy(predicted_answer, correct_answer_idx)

                return {
                    "final_answer_text": model_output_result,
                    "predicted_answer": predicted_answer,
                    "correct_answer": correct_answer,
                    "correct_answer_idx": correct_answer_idx,
                    "accuracy": accuracy,
                    "original_predicted_answer": original_predicted_answer,
                    "original_accuracy": original_accuracy,
                    "generated_image_path": generated_image_path,
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
                        "original_predicted_answer": original_predicted_answer,
                        "original_accuracy": original_accuracy,
                        "generated_image_path": generated_image_path,
                        "error": str(e),
                    }

        return {
            "final_answer_text": "",
            "predicted_answer": None,
            "correct_answer": correct_answer,
            "correct_answer_idx": correct_answer_idx,
            "accuracy": 0.0,
            "original_predicted_answer": original_predicted_answer,
            "original_accuracy": original_accuracy,
            "generated_image_path": generated_image_path,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Re-inference: feed pre-generated thinking image back as 3rd input and answer directly."
    )
    parser.add_argument(
        "--results_file",
        required=True,
        help="Path to prior inference results JSON (merged or shard) containing saved_image_paths.",
    )
    parser.add_argument(
        "--data_dir",
        default="/path/to/scratch/spatial_collab_dataset",
        help="Original dataset directory (needed to look up raw image paths by sample_id).",
    )
    parser.add_argument(
        "--dataset_files",
        nargs="+",
        default=["spatial_dataset_V_Final_2000.json"],
        help="Original dataset JSON file(s) inside data_dir.",
    )
    parser.add_argument(
        "--model_path",
        default="/path/to/scratch/models/BAGEL-7B-MoT",
    )
    parser.add_argument("--max_mem_per_gpu", default="80GiB")
    parser.add_argument("--vit_min_size", type=int, default=512)
    parser.add_argument(
        "--thinking_mode",
        choices=["no_thinking", "text_only_thinking", "understanding"],
        default="no_thinking",
    )
    parser.add_argument(
        "--output_file",
        default="/path/to/scratch/spatial_collab/BAGEL/rethink_results.json",
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help="Shard in format 'index/total' (e.g., '0/4').",
    )
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
    parser.add_argument("--image_shapes", type=int, nargs=2, default=[320, 1024])
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--no_resume", action="store_true")

    args = parser.parse_args()
    set_seed(args.random_seed)

    # Load prior results (provides sample_id, question, options, saved_image_paths, etc.)
    print(f"Loading prior results from {args.results_file}")
    with open(args.results_file, "r") as f:
        prior_data = json.load(f)
    prior_results = prior_data.get("results", [])
    print(f"Loaded {len(prior_results)} prior results")

    # Filter to samples that have at least one saved generated image
    prior_results = [r for r in prior_results if r.get("saved_image_paths")]
    print(f"Samples with saved generated images: {len(prior_results)}")

    # Load original dataset to get raw image paths (user_1_image_local_path, user_2_image_local_path)
    sample_id_to_raw = {}
    for dataset_file in args.dataset_files:
        file_path = os.path.join(args.data_dir, dataset_file)
        print(f"Loading dataset from {file_path}")
        with open(file_path, "r") as f:
            raw_data = json.load(f)
        for example in raw_data:
            sid = example.get("sample_id", "")
            if sid:
                sample_id_to_raw[sid] = example
    print(f"Indexed {len(sample_id_to_raw)} samples from original dataset")

    # Only keep prior_results that have a matching raw entry
    prior_results = [r for r in prior_results if r.get("sample_id", "") in sample_id_to_raw]
    print(f"Samples with both generated image and raw data: {len(prior_results)}")

    if args.num_samples is not None and args.num_samples < len(prior_results):
        random.seed(args.random_seed)
        prior_results = random.sample(prior_results, args.num_samples)
        print(f"Randomly sampled {args.num_samples} samples")

    # Apply sharding
    if args.shard is not None:
        try:
            shard_idx, total_shards = map(int, args.shard.split("/"))
            prior_results = sorted(prior_results, key=lambda x: x.get("sample_id", ""))
            total_samples = len(prior_results)
            shard_size = total_samples // total_shards
            remainder = total_samples % total_shards
            start_idx = shard_idx * shard_size + min(shard_idx, remainder)
            end_idx = start_idx + shard_size + (1 if shard_idx < remainder else 0)
            prior_results = prior_results[start_idx:end_idx]
            print(f"Shard {shard_idx}/{total_shards}: {len(prior_results)} samples")
        except ValueError as e:
            print(f"Error parsing shard argument '{args.shard}': {e}")
            return

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    checkpoint_path = args.output_file.replace(".json", "_checkpoint.json")
    checkpoint_mgr = CheckpointManager(checkpoint_path, save_interval=args.checkpoint_interval)

    completed_sample_ids = set()
    if not args.no_resume:
        if os.path.exists(checkpoint_path):
            checkpoint_mgr.load_checkpoint()
            completed_sample_ids = checkpoint_mgr.get_completed_sample_ids()
            print(f"Resuming: {len(completed_sample_ids)} done, {len(prior_results) - len(completed_sample_ids)} remaining")
        elif os.path.exists(args.output_file):
            try:
                with open(args.output_file, "r") as f:
                    output_data = json.load(f)
                for r in output_data.get("results", []):
                    sid = r.get("sample_id", "")
                    if sid and r.get("final_answer_text", "").strip():
                        checkpoint_mgr.results[sid] = r
                completed_sample_ids = checkpoint_mgr.get_completed_sample_ids()
                print(f"Loaded {len(completed_sample_ids)} completed from output file")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load output file: {e}")

    # Initialize inference engine
    inference_engine = BAGELRethinkInference(
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

    def load_images_for_sample(prior_result: Dict[str, Any]):
        sample_id = prior_result["sample_id"]
        raw = sample_id_to_raw[sample_id]

        user_1_path = raw["user_1_image_local_path"]
        user_2_path = raw["user_2_image_local_path"]

        # Handle relative paths or path substitution (same logic as run_inference_bagel_spatial.py)
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

        answerer_image = Image.open(user_1_image_path).convert('RGB')
        helper_image = Image.open(user_2_image_path).convert('RGB')
        generated_image_path = prior_result["saved_image_paths"][0]
        generated_image = Image.open(generated_image_path).convert('RGB')

        return answerer_image, helper_image, generated_image, generated_image_path

    def process_single_sample(prior_result: Dict[str, Any]):
        sample_id = prior_result.get("sample_id", "")
        try:
            answerer_image, helper_image, generated_image, generated_image_path = load_images_for_sample(prior_result)

            result = inference_engine.run_single_sample(
                answerer_image=answerer_image,
                helper_image=helper_image,
                generated_image=generated_image,
                question=prior_result["question"],
                options=prior_result["options"],
                correct_answer_idx=prior_result["correct_answer_idx"],
                correct_answer=prior_result["correct_answer"],
                question_type=prior_result.get("question_type", ""),
                thinking_mode=args.thinking_mode,
                sample_id=sample_id,
                original_predicted_answer=prior_result.get("predicted_answer"),
                original_accuracy=prior_result.get("accuracy", 0.0),
                generated_image_path=generated_image_path,
            )

            result["sample_id"] = sample_id
            result["question_type"] = prior_result.get("question_type", "")
            result["question"] = prior_result["question"]
            result["options"] = prior_result["options"]
            return result

        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    samples_to_process = [
        r for r in prior_results if r.get("sample_id", "") not in completed_sample_ids
    ]

    print(f"\nRunning rethink inference...")
    print(f"Samples to process: {len(samples_to_process)}")
    print(f"Thinking mode: {args.thinking_mode}")

    try:
        for prior_result in tqdm(samples_to_process, desc="Running rethink inference"):
            result = process_single_sample(prior_result)
            if result:
                checkpoint_mgr.add_result(result)

            all_results = checkpoint_mgr.get_all_results()
            if len(all_results) > 0 and len(all_results) % 10 == 0:
                current_acc = sum(r["accuracy"] for r in all_results) / len(all_results)
                orig_acc = sum(r.get("original_accuracy", 0.0) for r in all_results) / len(all_results)
                print(f"Current accuracy: {current_acc:.4f} | Original accuracy: {orig_acc:.4f} ({len(all_results)} samples)")

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        checkpoint_mgr.save_final()
        return

    checkpoint_mgr.save_final()
    results = checkpoint_mgr.get_all_results()

    final_accuracy = sum(r["accuracy"] for r in results) / len(results) if results else 0.0
    orig_accuracy = sum(r.get("original_accuracy", 0.0) for r in results) / len(results) if results else 0.0

    print(f"\n{'='*60}")
    print(f"Rethink Inference Complete!")
    print(f"Total Samples: {len(results)}")
    print(f"Rethink Accuracy:  {final_accuracy:.4f}")
    print(f"Original Accuracy: {orig_accuracy:.4f}")
    print(f"{'='*60}")

    output_data = {
        "config": {
            "model_path": args.model_path,
            "results_file": args.results_file,
            "thinking_mode": args.thinking_mode,
            "num_samples": len(results),
            "text_temperature": args.text_temperature,
            "shard": args.shard,
        },
        "metrics": {
            "rethink_accuracy": final_accuracy,
            "original_accuracy": orig_accuracy,
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
