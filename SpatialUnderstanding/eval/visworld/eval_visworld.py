#!/usr/bin/env python
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
VisWorld-Eval Evaluation Script

Evaluates ThinkMorph on VisWorld-Eval (7 tasks: ballgame, cube, maze, mmsi,
multihop, paperfolding, sokoban).

Usage:
    # Run inference on all splits
    python eval_visworld.py \
        --model_path /path/to/ThinkMorph-7B \
        --output_dir ./visworld_results \
        --num_passes 2

    # Run inference on specific splits
    python eval_visworld.py \
        --model_path /path/to/ThinkMorph-7B \
        --splits ballgame cube \
        --output_dir ./visworld_results

    # Run evaluation with LLM judge
    python eval_visworld.py \
        --mode evaluate \
        --results_dir ./visworld_results \
        --output_dir ./visworld_evaluated
"""

import os
import io
import sys
import json
import argparse
import regex
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import pandas as pd
from PIL import Image
from datasets import load_dataset
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from google import genai
from multiprocessing import Pool

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer


# ============================================================================
# Constants
# ============================================================================

ALL_SPLITS = ["ballgame", "cube", "maze", "mmsi", "multihop", "paperfolding", "sokoban"]

# Splits where the answer is an integer (not a string)
INTEGER_ANSWER_SPLITS = {"ballgame", "paperfolding"}


# ============================================================================
# LLM Judge Configuration
# ============================================================================

LLM_JUDGE_PROMPT = """You are evaluating whether a model's answer matches the ground truth for a visual reasoning question.

Question: {question}
Ground Truth Answer: {groundtruth}
Model's Extracted Answer: {modeloutput}

Determine if the model's answer is equivalent to the ground truth. Consider:
- For multiple choice questions, the letter (A, B, C, D) should match
- For numerical answers, the value should match (allow minor formatting differences)
- For text answers, the meaning should be equivalent
- For spatial reasoning tasks, focus on whether the core spatial relationship or answer is correct
- For path/sequence answers, check if the waypoints or steps are equivalent

Respond with ONLY "True" if the model's answer matches the ground truth, or "False" if it does not match."""

# Global client for multiprocessing
client_judge = None
JUDGE_MODEL_NAME = None


def init_worker(api_key, base_url, model_name):
    """Initialize the Gemini client for each worker process."""
    global client_judge, JUDGE_MODEL_NAME
    if api_key:
        os.environ['VisualCoT_GEMINI'] = api_key
    client_judge = genai.Client()
    JUDGE_MODEL_NAME = model_name


def judge_single_result(item):
    """Judge a single result using Gemini LLM."""
    global client_judge, JUDGE_MODEL_NAME
    NUM_RETRIES = 3

    for retry in range(NUM_RETRIES):
        try:
            question = item.get('Question', '')
            groundtruth = item.get('GroundTruth', '')
            extracted = item.get('ExtractedAnswer', '')

            if not extracted:
                return {**item, 'LLMJudgeResult': False}

            response = client_judge.models.generate_content(
                model=JUDGE_MODEL_NAME,
                contents=LLM_JUDGE_PROMPT.format(
                    question=question,
                    groundtruth=groundtruth,
                    modeloutput=extracted
                )
            )

            judge_response = str(response.text).strip().lower()
            llm_judge_result = 'true' in judge_response

            return {**item, 'LLMJudgeResult': llm_judge_result}

        except Exception as e:
            if retry == NUM_RETRIES - 1:
                print(f"Error judging ID {item.get('Id')}: {e}")
                return {**item, 'LLMJudgeResult': False, 'JudgeError': str(e)}

    return {**item, 'LLMJudgeResult': False}


# ============================================================================
# Model Loading
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_path: str, max_mem_per_gpu: str = "80GiB"):
    """Load the ThinkMorph model and prepare all components."""
    print(f"Loading model from {model_path}...")

    # LLM config
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # Bagel config
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

    # Tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Image transforms
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # Device map
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    print(f"Device map: {device_map}")

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
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    # Load checkpoint
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
        offload_folder="/tmp/offload"
    )

    model = model.eval()
    print('Model loaded successfully')

    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


# ============================================================================
# Data Loading
# ============================================================================

def load_visworld_data(
    data_source: str = "parquet",
    parquet_path: Optional[str] = None,
    splits: Optional[List[str]] = None,
    image_dir: str = "./visworld_images",
    cache_dir: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load VisWorld-Eval data from local parquet files or HuggingFace.

    Args:
        data_source: "huggingface" or "parquet"
        parquet_path: Root path containing split subdirectories (if parquet)
        splits: List of splits to load (default: all 7)
        image_dir: Directory to save extracted images
        cache_dir: HuggingFace cache directory

    Returns:
        Dict mapping split name -> list of data items
    """
    if splits is None:
        splits = ALL_SPLITS

    os.makedirs(image_dir, exist_ok=True)

    all_split_data = {}

    for split_name in splits:
        print(f"\nLoading split: {split_name}")

        if data_source == "huggingface":
            dataset = load_dataset(
                "thuml/VisWorld-Eval",
                split=split_name,
                cache_dir=cache_dir,
            )
            df = dataset.to_pandas()
        else:
            parquet_file = os.path.join(parquet_path, split_name, f"{split_name}.parquet")
            if not os.path.exists(parquet_file):
                # Try to find any parquet file in the split directory
                split_dir = os.path.join(parquet_path, split_name)
                if os.path.isdir(split_dir):
                    for f in os.listdir(split_dir):
                        if f.endswith('.parquet'):
                            parquet_file = os.path.join(split_dir, f)
                            break
            print(f"  Loading from: {parquet_file}")
            df = pd.read_parquet(parquet_file)

        print(f"  Loaded {len(df)} samples")

        data = []
        split_image_dir = os.path.join(image_dir, split_name)
        os.makedirs(split_image_dir, exist_ok=True)

        for i, (idx, row) in enumerate(df.iterrows()):
            if i % 50 == 0:
                print(f"  Processing {split_name}: {i}/{len(df)}", flush=True)
            index_val = row['index']
            prompt = row['prompt']
            answer = row['answer']
            category = row.get('category', split_name)
            # mmsi split has an extra question_type column
            question_type = row.get('question_type', '')
            # sokoban has a board column
            board = row.get('board', '')

            # Extract and save images
            image_paths = []
            images_data = row.get('image', None)
            if images_data is not None:
                for n, img_entry in enumerate(images_data):
                    img_bytes = img_entry['bytes']
                    image_path = os.path.join(split_image_dir, f"{index_val}_{n}.png")
                    if not os.path.exists(image_path):
                        with open(image_path, "wb") as f:
                            f.write(img_bytes)
                    image_paths.append(image_path)

            # Format prompt with boxed answer instruction
            formatted_prompt = prompt
            formatted_prompt += "\nThink about the question and give your final answer in \\boxed{Answer} format."

            # Convert answer to string for consistency
            answer_str = str(answer)

            data.append({
                "Id": f"{split_name}_{index_val}",
                "Index": index_val,
                "question": formatted_prompt,
                "original_question": prompt,
                "image_paths": image_paths,
                "answer": answer_str,
                "Category": category,
                "Split": split_name,
                "QuestionType": question_type,
                "Board": board,
            })

        all_split_data[split_name] = data

    return all_split_data


# ============================================================================
# Inference
# ============================================================================

def extract_boxed_answer(text):
    """
    Extract the content from the last \\boxed{} pattern.
    Also supports alternative format: <answer>...</answer>
    """
    if text is None:
        return None

    # Match \boxed{...} with support for nested braces
    pattern = r'\\boxed\{((?:[^{}]|{(?:[^{}]|{.*})*})*)\}'
    matches = regex.findall(pattern, text)

    if matches:
        return matches[-1]

    # Alternative pattern for <answer>...</answer>
    pattern_alt = r'<answer>(.*?)</answer>'
    matches_alt = regex.findall(pattern_alt, text, regex.DOTALL)
    if matches_alt:
        return matches_alt[-1].strip()

    return None


def process_single_sample(
    example: Dict[str, Any],
    inferencer: InterleaveInferencer,
    inference_hyper: Dict[str, Any],
    output_dir: str,
    pass_num: int,
) -> Dict[str, Any]:
    """Process a single VisWorld-Eval sample with the inferencer."""
    NUM_RETRIES = 3

    for retry in range(NUM_RETRIES):
        try:
            task_id = example['Id']
            question = example['question']
            image_paths = example['image_paths']
            answer = example['answer']

            # Build interleaved input list: [image1, image2, ..., text]
            images = [Image.open(p) for p in image_paths] if image_paths else []

            if len(images) <= 1:
                # Single image (or no image): use the simple API
                output_list = inferencer(
                    image=images[0] if images else None,
                    text=question,
                    understanding_output=False,
                    think=True,
                    **inference_hyper
                )
            else:
                # Multiple images: pass as interleaved input_list
                input_list = images + [question]
                output_list = inferencer(
                    input_list=input_list,
                    understanding_output=False,
                    think=True,
                    **inference_hyper
                )

            # Process outputs
            model_output_result = ""
            generated_images = []
            text_round = 0

            for out_item in output_list:
                if isinstance(out_item, str):
                    model_output_result += f"[Round {text_round}]\n{out_item}\n\n"
                    text_round += 1
                elif isinstance(out_item, Image.Image):
                    image_save_dir = os.path.join(
                        output_dir, f"pass_{pass_num}", f"task_{task_id}"
                    )
                    os.makedirs(image_save_dir, exist_ok=True)
                    img_filename = f"generated_round_{len(generated_images)}.png"
                    img_path = os.path.join(image_save_dir, img_filename)
                    out_item.save(img_path)
                    generated_images.append(img_path)
                    model_output_result += f"[Generated Image {len(generated_images)}: {img_path}]\n\n"

            model_output_result = model_output_result.strip()
            extracted_answer = extract_boxed_answer(model_output_result)

            return {
                "Id": task_id,
                "Index": example['Index'],
                "Question": question,
                "OriginalQuestion": example['original_question'],
                "ModelResult": model_output_result,
                "GroundTruth": answer,
                "ExtractedAnswer": extracted_answer if extracted_answer else "",
                "LLMJudgeResult": False,
                "Split": example['Split'],
                "Category": example['Category'],
                "QuestionType": example.get('QuestionType', ''),
                "ImagePaths": image_paths,
                "GeneratedImages": generated_images,
            }

        except Exception as e:
            if retry < NUM_RETRIES - 1:
                print(f"Error processing {example['Id']}: {e}. Retrying ({retry+1}/{NUM_RETRIES})...")
            else:
                print(f"Error processing {example['Id']}: {e}. All retries failed.")

    return {
        "Id": example['Id'],
        "Index": example.get('Index', -1),
        "Question": example.get('question', ''),
        "OriginalQuestion": example.get('original_question', ''),
        "ModelResult": "",
        "GroundTruth": example.get('answer', ''),
        "ExtractedAnswer": "",
        "LLMJudgeResult": False,
        "Error": str(e),
        "Split": example.get('Split', ''),
        "Category": example.get('Category', ''),
        "QuestionType": example.get('QuestionType', ''),
        "ImagePaths": example.get('image_paths', []),
        "GeneratedImages": [],
    }


# ============================================================================
# Evaluation
# ============================================================================

def process_results_file(input_path, output_path, api_key, base_url, model_name, num_processes):
    """Process a single results file with LLM judge."""
    print(f"Processing: {input_path}")

    with open(input_path, 'r') as f:
        results = json.load(f)

    print(f"  Loaded {len(results)} results")

    with Pool(
        processes=num_processes,
        initializer=init_worker,
        initargs=(api_key, base_url, model_name)
    ) as pool:
        judged_results = []
        for i, result in enumerate(pool.imap(judge_single_result, results)):
            judged_results.append(result)
            if (i + 1) % 10 == 0 or (i + 1) == len(results):
                print(f"  Judging: {i+1}/{len(results)}", flush=True)

    with open(output_path, 'w') as f:
        json.dump(judged_results, f, indent=2)

    # Overall stats
    correct = sum(1 for r in judged_results if r.get('LLMJudgeResult'))
    total = len(judged_results)
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"  Results: {correct}/{total} correct ({accuracy:.2f}%)")
    print(f"  Saved to: {output_path}")

    # Per-split stats
    split_stats = {}
    for r in judged_results:
        s = r.get('Split', 'unknown')
        if s not in split_stats:
            split_stats[s] = {'correct': 0, 'total': 0}
        split_stats[s]['total'] += 1
        if r.get('LLMJudgeResult'):
            split_stats[s]['correct'] += 1

    print("\n  Per-split accuracy:")
    for s, stats in sorted(split_stats.items()):
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"    {s}: {stats['correct']}/{stats['total']} ({acc:.2f}%)")

    # Per question type (for mmsi split)
    qtype_stats = {}
    for r in judged_results:
        qt = r.get('QuestionType', '')
        if qt:
            if qt not in qtype_stats:
                qtype_stats[qt] = {'correct': 0, 'total': 0}
            qtype_stats[qt]['total'] += 1
            if r.get('LLMJudgeResult'):
                qtype_stats[qt]['correct'] += 1

    if qtype_stats:
        print("\n  Per-question-type accuracy:")
        for qt, stats in sorted(qtype_stats.items()):
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"    {qt}: {stats['correct']}/{stats['total']} ({acc:.2f}%)")

    return correct, total, split_stats, qtype_stats


def run_evaluation(args):
    """Run LLM judge evaluation on inference results."""
    api_key = args.api_key or os.environ.get('VisualCoT_GEMINI')
    if not api_key:
        print("ERROR: API key required. Use --api_key or set VisualCoT_GEMINI env var.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("VisWorld-Eval Evaluation")
    print("=" * 60)
    print(f"Judge Model: {args.judge_model}")
    print(f"Results Dir: {args.results_dir}")
    print(f"Output Dir:  {args.output_dir}")
    print(f"Processes:   {args.num_processes}")
    print("=" * 60)
    print()

    # Find result files
    result_files = []
    for filename in sorted(os.listdir(args.results_dir)):
        if filename.startswith('model_results_run_') and filename.endswith('.json'):
            result_files.append(filename)

    if not result_files:
        print("No result files found!")
        return 1

    print(f"Found {len(result_files)} result files: {result_files}")
    print()

    all_stats = []
    all_split_stats = {}

    for filename in result_files:
        input_path = os.path.join(args.results_dir, filename)
        output_path = os.path.join(args.output_dir, f"evaluated_{filename}")

        correct, total, split_stats, _ = process_results_file(
            input_path, output_path,
            api_key, args.base_url, args.judge_model,
            args.num_processes
        )
        all_stats.append({'file': filename, 'correct': correct, 'total': total})

        for s, stats in split_stats.items():
            if s not in all_split_stats:
                all_split_stats[s] = {'correct': 0, 'total': 0}
            all_split_stats[s]['correct'] += stats['correct']
            all_split_stats[s]['total'] += stats['total']

        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    total_correct = sum(s['correct'] for s in all_stats)
    total_samples = sum(s['total'] for s in all_stats)

    for s in all_stats:
        acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
        print(f"  {s['file']}: {s['correct']}/{s['total']} ({acc:.2f}%)")

    if len(all_stats) > 1:
        avg_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
        print(f"\n  Overall: {total_correct}/{total_samples} ({avg_acc:.2f}%)")

    # Per-split summary across all passes
    print("\n  Per-split (aggregated):")
    # Compute 5-task and 7-task averages matching the paper's leaderboard
    split_accuracies = {}
    for s, stats in sorted(all_split_stats.items()):
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        split_accuracies[s] = acc
        print(f"    {s}: {stats['correct']}/{stats['total']} ({acc:.2f}%)")

    five_task_splits = {"ballgame", "cube", "mmsi", "multihop", "paperfolding"}
    five_task_accs = [split_accuracies[s] for s in five_task_splits if s in split_accuracies]
    all_task_accs = list(split_accuracies.values())
    if five_task_accs:
        print(f"\n  Overall (5 tasks, excl. maze & sokoban): {np.mean(five_task_accs):.2f}%")
    if all_task_accs:
        print(f"  Overall (7 tasks): {np.mean(all_task_accs):.2f}%")

    # Save summary
    summary = {
        'results': all_stats,
        'total_correct': total_correct,
        'total_samples': total_samples,
        'accuracy': total_correct / total_samples if total_samples > 0 else 0,
        'per_split': {k: {**v, 'accuracy': v['correct']/v['total'] if v['total'] > 0 else 0}
                      for k, v in all_split_stats.items()},
    }

    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    return 0


def run_inference(args):
    """Run model inference on VisWorld-Eval."""
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("VisWorld-Eval Inference with ThinkMorph")
    print("=" * 60)

    # Load model
    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(
        args.model_path, args.max_mem_per_gpu
    )

    # Create inferencer
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids
    )

    # Inference hyperparameters
    inference_hyper = {
        'max_think_token_n': args.max_think_token_n,
        'do_sample': args.do_sample,
        'text_temperature': args.text_temperature,
        'cfg_text_scale': args.cfg_text_scale,
        'cfg_img_scale': args.cfg_img_scale,
        'cfg_interval': [args.cfg_interval_start, args.cfg_interval_end],
        'timestep_shift': args.timestep_shift,
        'num_timesteps': args.num_timesteps,
        'cfg_renorm_min': args.cfg_renorm_min,
        'cfg_renorm_type': args.cfg_renorm_type,
    }

    # Load data
    image_dir = os.path.join(args.output_dir, "images")
    splits = args.splits if args.splits else None
    all_split_data = load_visworld_data(
        data_source=args.data_source,
        parquet_path=args.parquet_path,
        splits=splits,
        image_dir=image_dir,
        cache_dir=args.cache_dir,
    )

    # Flatten all splits into one list for inference
    all_data = []
    for split_name, split_data in all_split_data.items():
        all_data.extend(split_data)

    # Shard data if running in parallel mode
    total_before_shard = len(all_data)
    if args.num_shards > 1:
        shard_size = len(all_data) // args.num_shards
        remainder = len(all_data) % args.num_shards
        start = args.shard_id * shard_size + min(args.shard_id, remainder)
        end = start + shard_size + (1 if args.shard_id < remainder else 0)
        all_data = all_data[start:end]

    print(f"\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  Splits: {list(all_split_data.keys())}")
    print(f"  Total samples: {total_before_shard}")
    if args.num_shards > 1:
        print(f"  Shard: {args.shard_id}/{args.num_shards} ({len(all_data)} samples)")
    for split_name, split_data in all_split_data.items():
        print(f"    {split_name}: {len(split_data)}")
    print(f"  Passes: {args.num_passes}")
    print()

    # Run passes
    for pass_num in range(args.num_passes):
        # Determine output and checkpoint paths for this pass
        if args.num_shards > 1:
            output_path = os.path.join(
                args.output_dir, f"model_results_run_{pass_num + 1}_shard_{args.shard_id}.json"
            )
            ckpt_path = os.path.join(
                args.output_dir, f"checkpoint_run_{pass_num + 1}_shard_{args.shard_id}.jsonl"
            )
        else:
            output_path = os.path.join(args.output_dir, f"model_results_run_{pass_num + 1}.json")
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_run_{pass_num + 1}.jsonl")

        # If final output already exists, skip this pass entirely
        if os.path.exists(output_path):
            print(f"Pass {pass_num + 1}: output already exists at {output_path}, skipping.")
            continue

        # Resume from checkpoint if available
        results = []
        completed_ids = set()
        if os.path.exists(ckpt_path):
            with open(ckpt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        results.append(r)
                        completed_ids.add(r['Id'])
            print(f"Pass {pass_num + 1}: resumed from checkpoint, "
                  f"{len(results)}/{len(all_data)} already done.")

        remaining = [ex for ex in all_data if ex['Id'] not in completed_ids]
        print(f"Starting pass {pass_num + 1}/{args.num_passes} "
              f"({len(remaining)} remaining, {len(completed_ids)} cached)...")

        # Open checkpoint file in append mode for incremental saves
        with open(ckpt_path, 'a') as ckpt_f:
            for i, example in enumerate(remaining):
                print(f"Pass {pass_num + 1}: {i+1}/{len(remaining)}  "
                      f"(id={example['Id']})", flush=True)
                result = process_single_sample(
                    example,
                    inferencer,
                    inference_hyper,
                    args.output_dir,
                    pass_num + 1,
                )
                results.append(result)
                # Append result to checkpoint immediately
                ckpt_f.write(json.dumps(result) + '\n')
                ckpt_f.flush()

        # Write final consolidated output
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # Clean up checkpoint file after successful completion
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        # Statistics
        successful = sum(1 for r in results if 'Error' not in r)
        has_answer = sum(1 for r in results if r.get('ExtractedAnswer'))
        has_generated_images = sum(1 for r in results if r.get('GeneratedImages'))

        print(f"\n{'=' * 50}")
        print(f"Pass {pass_num + 1} Complete")
        print(f"{'=' * 50}")
        print(f"Total: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Has Extracted Answer: {has_answer}")
        print(f"Tasks with Generated Images: {has_generated_images}")

        # Per-split breakdown
        split_counts = {}
        for r in results:
            s = r.get('Split', 'unknown')
            if s not in split_counts:
                split_counts[s] = {'total': 0, 'successful': 0, 'has_answer': 0}
            split_counts[s]['total'] += 1
            if 'Error' not in r:
                split_counts[s]['successful'] += 1
            if r.get('ExtractedAnswer'):
                split_counts[s]['has_answer'] += 1

        for s, counts in sorted(split_counts.items()):
            print(f"  {s}: {counts['successful']}/{counts['total']} successful, "
                  f"{counts['has_answer']} with extracted answer")

        print(f"Results saved to: {output_path}")
        print(f"{'=' * 50}\n")

    print(f"\nAll passes complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"\nRun with --mode evaluate to compute accuracy scores with LLM judge.")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='VisWorld-Eval Evaluation with ThinkMorph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run inference on all splits from local parquet
    python eval_visworld.py \\
        --mode inference \\
        --model_path /path/to/ThinkMorph-7B \\
        --data_source parquet \\
        --parquet_path /path/to/VisWorld-Eval \\
        --output_dir ./visworld_results

    # Run inference on specific splits
    python eval_visworld.py \\
        --mode inference \\
        --model_path /path/to/ThinkMorph-7B \\
        --splits ballgame cube mmsi \\
        --output_dir ./visworld_results

    # Run evaluation with LLM judge
    python eval_visworld.py \\
        --mode evaluate \\
        --results_dir ./visworld_results \\
        --output_dir ./visworld_evaluated
"""
    )

    parser.add_argument('--mode', type=str, default='inference',
                        choices=['inference', 'evaluate'],
                        help='Mode: inference or evaluate (default: inference)')

    # Data arguments
    parser.add_argument('--data_source', type=str, default='parquet',
                        choices=['huggingface', 'parquet'],
                        help='Data source: huggingface or parquet')
    parser.add_argument('--parquet_path', type=str,
                        default='/path/to/scratch/datasets/VisWorld-Eval',
                        help='Root path to VisWorld-Eval parquet data')
    parser.add_argument('--splits', type=str, nargs='+', default=None,
                        choices=ALL_SPLITS,
                        help='Splits to evaluate (default: all 7)')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='HuggingFace cache directory')

    # Model arguments
    parser.add_argument('--model_path', type=str,
                        default='/path/to/scratch/models/ThinkMorph-7B',
                        help='Path to the model directory')
    parser.add_argument('--max_mem_per_gpu', type=str, default='80GiB',
                        help='Maximum memory per GPU')

    # Output arguments
    parser.add_argument('--output_dir', type=str,
                        default='/path/to/scratch/VisualCoT/visworld_results',
                        help='Directory to save outputs')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory containing results (for evaluate mode)')

    # Inference arguments
    parser.add_argument('--num_passes', type=int, default=2,
                        help='Number of evaluation passes (default: 2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Data-parallel sharding
    parser.add_argument('--shard_id', type=int, default=0,
                        help='Shard index for data-parallel inference (0-indexed)')
    parser.add_argument('--num_shards', type=int, default=1,
                        help='Total number of shards (set >1 to enable data parallelism)')

    # Inference hyperparameters
    parser.add_argument('--max_think_token_n', type=int, default=4096)
    parser.add_argument('--do_sample', action='store_true', default=True)
    parser.add_argument('--text_temperature', type=float, default=0.3)
    parser.add_argument('--cfg_text_scale', type=float, default=4.0)
    parser.add_argument('--cfg_img_scale', type=float, default=2.0)
    parser.add_argument('--cfg_interval_start', type=float, default=0.0)
    parser.add_argument('--cfg_interval_end', type=float, default=1.0)
    parser.add_argument('--timestep_shift', type=float, default=3.0)
    parser.add_argument('--num_timesteps', type=int, default=50)
    parser.add_argument('--cfg_renorm_min', type=float, default=0.0)
    parser.add_argument('--cfg_renorm_type', type=str, default='text_channel',
                        choices=['global', 'channel', 'text_channel'])

    # Evaluation arguments
    parser.add_argument('--api_key', type=str, default=None,
                        help='Gemini API key (or set VisualCoT_GEMINI env var)')
    parser.add_argument('--base_url', type=str, default=None,
                        help='API base URL (not used for Gemini)')
    parser.add_argument('--judge_model', type=str, default='gemini-2.5-flash-preview-05-20',
                        help='Judge model name')
    parser.add_argument('--num_processes', type=int, default=4,
                        help='Number of parallel processes for judging')

    args = parser.parse_args()

    if args.mode == 'inference':
        return run_inference(args)
    else:
        if args.results_dir is None:
            args.results_dir = args.output_dir
        return run_evaluation(args)


if __name__ == '__main__':
    exit(main())
