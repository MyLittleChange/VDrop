#!/usr/bin/env python3
"""
Test BAGEL model on panorama-view generation task.

Uses samples WITHOUT panorama_path from the approved MCQA benchmark files as the test set,
ensuring no scene overlap with the training set (samples WITH panorama_path).

Supports three thinking modes (matching training):
  --thinking_mode interleaved_thinking  (default)
  --thinking_mode text_only_thinking
  --thinking_mode visual_only_thinking
"""

import argparse
import json
import os
import sys
import random

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer


# --- System prompts for each thinking mode (must match training) ---

INTERLEAVED_THINK_SYSTEM_PROMPT = (
    "Let's think step by step to answer the question. "
    "For text-based thinking, enclose the process within <think> </think>. "
    "For visual thinking, enclose the content within <image_start> </image_end>."
)

TEXT_ONLY_THINK_SYSTEM_PROMPT = (
    "Let's think step by step to answer the question. "
    "Enclose your thinking process within <think> </think> tags."
)

VISUAL_ONLY_THINK_SYSTEM_PROMPT = (
    "Think visually to answer the question. "
    "Enclose your visual thinking within <image_start> </image_end>."
)

THINKING_MODE_PROMPTS = {
    "interleaved_thinking": INTERLEAVED_THINK_SYSTEM_PROMPT,
    "text_only_thinking": TEXT_ONLY_THINK_SYSTEM_PROMPT,
    "visual_only_thinking": VISUAL_ONLY_THINK_SYSTEM_PROMPT,
}


DEFAULT_ANNOTATION_FILES = [
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def remap_path(path: str) -> str:
    """Remap paths from home directories to network scratch."""
    if path is None:
        return None
    path = path.replace(
        "/path/to/scratch",
        "/path/to/scratch",
    )
    path = path.replace(
        "/path/to/scratch",
        "/path/to/scratch",
    )
    return path


def load_model(model_path: str, max_mem_per_gpu: str = "80GiB"):
    """Load the BAGEL model and prepare all components."""
    from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

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
        connector_act="gelu_pytorch_tanh",
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
    vit_transform = ImageTransform(980, 224, 14)

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    max_memory = {i: max_mem_per_gpu for i in range(num_gpus)}
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
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


def load_test_samples(annotation_files: list) -> list:
    """Load test samples (without panorama_path) ensuring no scene overlap with train."""
    train_scenes = set()
    test_samples = []

    for filepath in annotation_files:
        print(f"Loading {filepath}")
        with open(filepath, 'r') as f:
            samples = json.load(f)

        with_p = [s for s in samples if s.get('panorama_path')]
        without_p = [s for s in samples if not s.get('panorama_path')]
        print(f"  {len(samples)} total, {len(with_p)} train, {len(without_p)} test candidates")

        train_scenes.update(s['scene_id'] for s in with_p)
        test_samples.extend(without_p)

    # Remove any test samples whose scene appears in training
    before = len(test_samples)
    test_samples = [s for s in test_samples if s['scene_id'] not in train_scenes]
    test_scenes = set(s['scene_id'] for s in test_samples)

    if before != len(test_samples):
        print(f"\nRemoved {before - len(test_samples)} test samples due to scene overlap with train")

    print(f"\nTest set: {len(test_samples)} samples, {len(test_scenes)} scenes")
    print(f"Train scenes: {len(train_scenes)}, Scene overlap: 0")

    return test_samples


def main():
    parser = argparse.ArgumentParser(
        description="Test BAGEL on panorama-view generation task"
    )
    parser.add_argument(
        "--annotation_files",
        nargs="+",
        default=DEFAULT_ANNOTATION_FILES,
        help="Paths to the approved MCQA JSON files (used to derive train/test split)",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Path to a pre-split test JSON file (e.g. test_samples.json). "
             "If provided, loads test samples directly and skips train/test splitting.",
    )
    parser.add_argument(
        "--model_path",
        default="/path/to/scratch/models/BAGEL-7B-MoT",
    )
    parser.add_argument("--max_mem_per_gpu", default="80GiB")
    parser.add_argument(
        "--output_dir",
        default="/path/to/scratch/spatial_collab/BAGEL/panorama_view_test",
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--thinking_mode",
        choices=["interleaved_thinking", "text_only_thinking", "visual_only_thinking"],
        default="interleaved_thinking",
        help="Thinking mode (must match training mode)",
    )
    parser.add_argument("--think", action="store_true", default=True)
    parser.add_argument("--no_think", action="store_true")
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help="Shard specification in format 'index/total' (e.g., '0/48' for first of 48 shards).",
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
    parser.add_argument("--cfg_renorm_type", type=str, default="text_channel")

    args = parser.parse_args()
    think = args.think and not args.no_think
    thinking_mode = args.thinking_mode
    system_prompt = THINKING_MODE_PROMPTS[thinking_mode]
    understanding_output = thinking_mode == "text_only_thinking"

    set_seed(args.random_seed)

    # Load test samples
    if args.test_file:
        print(f"Loading pre-split test file: {args.test_file}")
        with open(args.test_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} test samples")
    else:
        results = load_test_samples(args.annotation_files)

    # Filter samples with valid input image paths
    valid_results = []
    for r in results:
        img1_path = remap_path(r.get("user_1_image_local_path"))
        img2_path = remap_path(r.get("user_2_image_local_path"))
        if img1_path and img2_path and os.path.exists(img1_path) and os.path.exists(img2_path):
            valid_results.append(r)
    results = valid_results
    print(f"After filtering missing images: {len(results)} samples")

    if args.num_samples is not None and args.num_samples < len(results):
        results = random.sample(results, args.num_samples)
        print(f"Randomly sampled {args.num_samples} samples")

    # Apply sharding if specified
    shard_idx = None
    if args.shard is not None:
        try:
            shard_idx, total_shards = map(int, args.shard.split("/"))
            if shard_idx < 0 or shard_idx >= total_shards:
                raise ValueError(f"Shard index {shard_idx} must be in range [0, {total_shards})")

            results = sorted(results, key=lambda x: x.get("sample_id", ""))

            total_samples = len(results)
            shard_size = total_samples // total_shards
            remainder = total_samples % total_shards

            start_idx = shard_idx * shard_size + min(shard_idx, remainder)
            end_idx = start_idx + shard_size + (1 if shard_idx < remainder else 0)

            results = results[start_idx:end_idx]
            print(f"Shard {shard_idx}/{total_shards}: processing samples {start_idx}-{end_idx-1} ({len(results)} samples)")
        except ValueError as e:
            print(f"Error parsing shard argument '{args.shard}': {e}")
            return

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    generated_dir = os.path.join(args.output_dir, "generated")
    input_dir = os.path.join(args.output_dir, "inputs")
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)

    # Load model
    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(
        args.model_path, args.max_mem_per_gpu
    )
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    inference_hyper = {
        "max_think_token_n": args.max_think_token_n,
        "do_sample": args.do_sample,
        "text_temperature": args.text_temperature,
        "cfg_text_scale": args.cfg_text_scale,
        "cfg_img_scale": args.cfg_img_scale,
        "cfg_interval": [args.cfg_interval_start, args.cfg_interval_end],
        "timestep_shift": args.timestep_shift,
        "num_timesteps": args.num_timesteps,
        "cfg_renorm_min": args.cfg_renorm_min,
        "cfg_renorm_type": args.cfg_renorm_type,
    }

    # Run inference
    output_records = []
    print(f"\nRunning inference on {len(results)} samples (thinking_mode={thinking_mode}, think={think})...")

    for idx, sample in enumerate(tqdm(results, desc="Generating panorama views")):
        sample_id = sample.get("sample_id", f"sample_{idx}")

        image1_path = remap_path(sample.get("user_1_image_local_path"))
        image2_path = remap_path(sample.get("user_2_image_local_path"))

        instruction = system_prompt + "\nGenerate the panoramic view of the room. "

        try:
            img1 = Image.open(image1_path).convert("RGB")
            img2 = Image.open(image2_path).convert("RGB")

            input_list = [img1, img2, instruction]

            output_list = inferencer(
                input_list=input_list,
                understanding_output=understanding_output,
                think=think,
                **inference_hyper,
            )

            # Collect generated images and text from output
            generated_images = []
            text_outputs = []
            for out_item in output_list:
                if isinstance(out_item, Image.Image):
                    generated_images.append(out_item)
                elif isinstance(out_item, str):
                    text_outputs.append(out_item)

            # Save the last generated image as the panorama prediction
            if generated_images:
                pred_image = generated_images[-1]
                pred_path = os.path.join(generated_dir, f"{sample_id}.png")
                pred_image.save(pred_path)

                for gi, gen_img in enumerate(generated_images):
                    gen_path = os.path.join(generated_dir, f"{sample_id}_round_{gi}.png")
                    gen_img.save(gen_path)
            else:
                pred_path = None
                print(f"[WARNING] No image generated for sample {sample_id}")

            # Save inputs for comparison
            img1_save = os.path.join(input_dir, f"{sample_id}_img1.png")
            img2_save = os.path.join(input_dir, f"{sample_id}_img2.png")
            img1.save(img1_save)
            img2.save(img2_save)

            record = {
                "sample_id": sample_id,
                "scene_id": sample.get("scene_id", ""),
                "instruction": instruction,
                "text_outputs": text_outputs,
                "generated_image_path": pred_path,
                "input_img1_path": img1_save,
                "input_img2_path": img2_save,
                "num_generated_images": len(generated_images),
            }
            output_records.append(record)

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(results)} samples, "
                      f"{sum(1 for r in output_records if r['generated_image_path'] is not None)} generated successfully")

        except Exception as e:
            print(f"[ERROR] Failed on sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            output_records.append({
                "sample_id": sample_id,
                "scene_id": sample.get("scene_id", ""),
                "instruction": instruction,
                "error": str(e),
                "generated_image_path": None,
            })

    # Save results summary
    summary = {
        "config": {
            "model_path": args.model_path,
            "thinking_mode": thinking_mode,
            "think": think,
            "num_samples": len(results),
            "inference_hyper": inference_hyper,
        },
        "metrics": {
            "total_samples": len(output_records),
            "successful_generations": sum(1 for r in output_records if r.get("generated_image_path") is not None),
            "failed": sum(1 for r in output_records if r.get("generated_image_path") is None),
        },
        "results": output_records,
    }

    shard_suffix = f"_shard{shard_idx}" if shard_idx is not None else ""
    summary_path = os.path.join(args.output_dir, f"results_summary{shard_suffix}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Panorama View Generation Test Complete!")
    print(f"Total samples: {summary['metrics']['total_samples']}")
    print(f"Successful generations: {summary['metrics']['successful_generations']}")
    print(f"Failed: {summary['metrics']['failed']}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
