#!/usr/bin/env python
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
MMSI-Bench Evaluation Script

Evaluates ThinkMorph on MMSI-Bench (Multi-Modal Spatial Intelligence Benchmark).

Usage:
    # Run inference
    python eval_mmsi_bench.py \
        --model_path /path/to/ThinkMorph-7B \
        --output_dir ./mmsi_results \
        --num_passes 3

    # Run evaluation with LLM judge
    python eval_mmsi_bench.py \
        --mode evaluate \
        --results_dir ./mmsi_results \
        --output_dir ./mmsi_evaluated
"""

import os
import json
import argparse
import regex
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from google import genai
from multiprocessing import Pool

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
from SpatialUnderstanding.dynamic_visual_tokens import add_dynamic_visual_tokens
from SpatialUnderstanding.dynamic_visual_inferencer import (
    DynamicVisualInterleaveInferencer,
    DYNAMIC_VISUAL_THINK_SYSTEM_PROMPT,
)


# ============================================================================
# Thinking Mode Prompts (matching run_inference_bagel_spatial.py)
# ============================================================================


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

TEXT_THINKING_SYSTEM_PROMPT = (
    "Think step by step before answering. "
    "Enclose your thinking within <think> </think> tags. "
    "Provide your final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)

NO_THINKING_SYSTEM_PROMPT = (
    "Answer the question directly. "
    "Provide your final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)

ZEBRA_COT_SYSTEM_PROMPT = (
    "You are an AI reasoning assistant capable of step-by-step interleaved text and visual chain of thought. "
    "Think step by step and generate visual aids to enhance your problem-solving. "
    "You should first think about the reasoning and planning process in the mind before generating visual aids. "
    "Wrap your text reasoning with <think></think> tokens, and wrap your final conclusion with <answer></answer> tokens. "
    "Provide your final conclusion clearly in the format of '<answer>Final Answer: <answer here></answer>'"
)

UNDERSTANDING_SYSTEM_PROMPT = (
    "You are given two camera views and a panoramic overview of the scene. "
    "Answer the question directly. "
    "Provide your final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)

THINKING_MODE_PROMPTS = {
    "dynamic_visual_thinking": DYNAMIC_VISUAL_THINK_SYSTEM_PROMPT,
    "visual_only_thinking": VISUAL_ONLY_THINK_SYSTEM_PROMPT,
    "interleaved_thinking": INTERLEAVED_THINK_SYSTEM_PROMPT,
    "text_only_thinking": TEXT_ONLY_THINK_SYSTEM_PROMPT,
    "text_thinking": TEXT_THINKING_SYSTEM_PROMPT,
    "no_thinking": NO_THINKING_SYSTEM_PROMPT,
    "zebra_cot": ZEBRA_COT_SYSTEM_PROMPT,
    "understanding": UNDERSTANDING_SYSTEM_PROMPT,
    "bagel_default": "",
}


# ============================================================================
# LLM Judge Configuration
# ============================================================================

LLM_JUDGE_PROMPT = """You are evaluating whether a model's answer matches the ground truth for a visual spatial intelligence question.

Question: {question}
Ground Truth Answer: {groundtruth}
Model's Extracted Answer: {modeloutput}

Determine if the model's answer is equivalent to the ground truth. Consider:
- For multiple choice questions, the letter (A, B, C, D) should match
- For numerical answers, the value should match (allow minor formatting differences)
- For text answers, the meaning should be equivalent
- For spatial reasoning tasks, focus on whether the core spatial relationship or answer is correct

Respond with ONLY "True" if the model's answer matches the ground truth, or "False" if it does not match."""

# Global client for multiprocessing
client_judge = None
JUDGE_MODEL_NAME = None


def init_worker(api_key, base_url, model_name):
    """Initialize the Gemini client for each worker process."""
    global client_judge, JUDGE_MODEL_NAME
    # Set API key via environment variable if provided
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

            if 'true' in judge_response:
                llm_judge_result = True
            else:
                llm_judge_result = False

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


def load_model(model_path: str, max_mem_per_gpu: str = "80GiB", vit_min_size: int = 512):
    """Load the ThinkMorph model and prepare all components."""
    print(f"Loading model from {model_path}...")

    # LLM config preparing
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config preparing
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # Bagel config preparing
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

    # Tokenizer Preparing
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    tokenizer, new_token_ids, _ = add_dynamic_visual_tokens(tokenizer, new_token_ids)

    # Image Transform Preparing
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, vit_min_size, 14)

    # Device map preparation
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

    # Load model checkpoint - support multiple checkpoint formats
    # Priority: model.safetensors > ema.safetensors > model directory
    checkpoint_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(checkpoint_path):
        # Try ema.safetensors (used by BAGEL-7B-MoT)
        checkpoint_path = os.path.join(model_path, "ema.safetensors")
        if not os.path.exists(checkpoint_path):
            # Fall back to model directory for auto-detection
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

def load_mmsi_bench_data(
    data_source: str = "huggingface",
    parquet_path: Optional[str] = None,
    image_dir: str = "./mmsi_images",
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load MMSI-Bench data from HuggingFace or parquet file.

    Args:
        data_source: "huggingface" or "parquet"
        parquet_path: Path to parquet file (if data_source is "parquet")
        image_dir: Directory to save extracted images
        cache_dir: HuggingFace cache directory

    Returns:
        List of data items with image paths and questions
    """
    os.makedirs(image_dir, exist_ok=True)

    if data_source == "huggingface":
        print("Loading MMSI-Bench from HuggingFace...")
        dataset = load_dataset("RunsenXu/MMSI-Bench", cache_dir=cache_dir)

        # Get the split (usually 'test' or 'train')
        if 'test' in dataset:
            data_split = dataset['test']
        elif 'train' in dataset:
            data_split = dataset['train']
        else:
            # Get first available split
            split_name = list(dataset.keys())[0]
            data_split = dataset[split_name]
            print(f"Using split: {split_name}")

        df = data_split.to_pandas()
    else:
        # Handle both directory path and direct file path
        if os.path.isdir(parquet_path):
            # Look for parquet file in directory
            parquet_file = os.path.join(parquet_path, "MMSI_Bench.parquet")
            if not os.path.exists(parquet_file):
                # Try alternative naming
                for f in os.listdir(parquet_path):
                    if f.endswith('.parquet'):
                        parquet_file = os.path.join(parquet_path, f)
                        break
        else:
            parquet_file = parquet_path

        print(f"Loading MMSI-Bench from parquet: {parquet_file}")
        df = pd.read_parquet(parquet_file)

    print(f"Loaded {len(df)} samples")

    # Process each record
    data = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing data"):
        id_val = row['id']
        images = row['images']
        question_type = row['question_type']
        question = row['question']
        answer = row['answer']
        thought = row.get('thought', '')
        difficulty = row.get('difficulty', 'unknown')
        mean_normed_duration = row.get('mean_normed_duration_seconds', 0)

        # Save images and collect paths
        image_paths = []
        if images is not None:
            for n, img_data in enumerate(images):
                image_path = os.path.join(image_dir, f"{id_val}_{n}.jpg")
                if not os.path.exists(image_path):
                    with open(image_path, "wb") as f:
                        f.write(img_data)
                image_paths.append(image_path)

        # Format the question with instruction
        formatted_question = question
        formatted_question += "\nThink about the question and give your final answer in \\boxed{Answer} format."

        data.append({
            "Id": id_val,
            "question": formatted_question,
            "original_question": question,
            "image_paths": image_paths,
            "answer": answer,
            "QuestionType": question_type,
            "Thought": thought,
            "Difficulty": difficulty,
            "MeanNormedDuration": mean_normed_duration,
        })

    return data


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
    think: bool = True,
    thinking_mode: str = "interleaved_thinking",
    understanding_output: bool = False,
) -> Dict[str, Any]:
    """Process a single MMSI-Bench sample with the inferencer."""
    NUM_RETRIES = 3

    for retry in range(NUM_RETRIES):
        try:
            task_id = example['Id']
            question = example['question']
            image_paths = example['image_paths']
            answer = example['answer']

            # Load image(s) - use first image for now
            # TODO: Handle multi-image inputs if needed
            if image_paths:
                image = Image.open(image_paths[0])
            else:
                # No image case
                image = None

            # Prepend thinking mode system prompt to question.
            # For "bagel_default", skip prepending so BAGEL's own internal default
            # system prompt is the only one in play (activated by think=True).
            system_prompt = THINKING_MODE_PROMPTS[thinking_mode]
            full_text = question if not system_prompt else system_prompt + "\n" + question

            # Run inference
            if thinking_mode == "zebra_cot":
                max_rounds = inference_hyper.get('max_rounds', 3)
                current_input = ([image] if image is not None else []) + [full_text]
                output_list = []
                for _ in range(max_rounds):
                    text_out = inferencer(
                        input_list=current_input,
                        understanding_output=True,
                        think=False,
                        **{k: v for k, v in inference_hyper.items() if k != 'max_rounds'}
                    )
                    text = text_out[0]
                    output_list.append(text)
                    current_input = current_input + [text]
                    if 'Final Answer:' in text or '<answer>' in text:
                        break
                    img_out = inferencer(
                        input_list=current_input,
                        understanding_output=False,
                        force_image_output=True,
                        think=False,
                        **{k: v for k, v in inference_hyper.items() if k != 'max_rounds'}
                    )
                    img = img_out[0]
                    output_list.append(img)
                    current_input = current_input + [img]
            else:
                output_list = inferencer(
                    image=image,
                    text=full_text,
                    understanding_output=understanding_output,
                    think=think,
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
                elif isinstance(out_item, tuple):
                    # (mode_name, Image.Image) from DynamicVisualInterleaveInferencer
                    mode_name, img = out_item
                    image_dir = os.path.join(output_dir, f"pass_{pass_num}", f"task_{task_id}")
                    os.makedirs(image_dir, exist_ok=True)
                    tag = mode_name.strip('<>')
                    img_filename = f"generated_round_{len(generated_images)}_{tag}.png"
                    img_path = os.path.join(image_dir, img_filename)
                    img.save(img_path)
                    generated_images.append(img_path)
                    model_output_result += f"[Generated Image {len(generated_images)} ({mode_name}): {img_path}]\n\n"
                elif isinstance(out_item, Image.Image):
                    image_dir = os.path.join(output_dir, f"pass_{pass_num}", f"task_{task_id}")
                    os.makedirs(image_dir, exist_ok=True)
                    img_filename = f"generated_round_{len(generated_images)}.png"
                    img_path = os.path.join(image_dir, img_filename)
                    out_item.save(img_path)
                    generated_images.append(img_path)
                    model_output_result += f"[Generated Image {len(generated_images)}: {img_path}]\n\n"

            model_output_result = model_output_result.strip()
            extracted_answer = extract_boxed_answer(model_output_result)

            return {
                "Id": task_id,
                "Question": question,
                "OriginalQuestion": example['original_question'],
                "ModelResult": model_output_result,
                "GroundTruth": answer,
                "ExtractedAnswer": extracted_answer if extracted_answer else "",
                "LLMJudgeResult": False,
                "QuestionType": example['QuestionType'],
                "Difficulty": example['Difficulty'],
                "Thought": example['Thought'],
                "ImagePaths": image_paths,
                "GeneratedImages": generated_images,
            }

        except Exception as e:
            if retry < NUM_RETRIES - 1:
                print(f"Error processing ID {example['Id']}: {e}. Retrying ({retry+1}/{NUM_RETRIES})...")
            else:
                print(f"Error processing ID {example['Id']}: {e}. All retries failed.")

    return {
        "Id": example['Id'],
        "Question": example.get('question', ''),
        "OriginalQuestion": example.get('original_question', ''),
        "ModelResult": "",
        "GroundTruth": example.get('answer', ''),
        "ExtractedAnswer": "",
        "LLMJudgeResult": False,
        "Error": str(e),
        "QuestionType": example.get('QuestionType', ''),
        "Difficulty": example.get('Difficulty', ''),
        "Thought": example.get('Thought', ''),
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
        judged_results = list(tqdm(
            pool.imap(judge_single_result, results),
            total=len(results),
            desc="  Judging"
        ))

    with open(output_path, 'w') as f:
        json.dump(judged_results, f, indent=2)

    # Calculate statistics
    correct = sum(1 for r in judged_results if r.get('LLMJudgeResult'))
    total = len(judged_results)
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"  Results: {correct}/{total} correct ({accuracy:.2f}%)")
    print(f"  Saved to: {output_path}")

    # Calculate per-type statistics
    type_stats = {}
    for r in judged_results:
        q_type = r.get('QuestionType', 'unknown')
        if q_type not in type_stats:
            type_stats[q_type] = {'correct': 0, 'total': 0}
        type_stats[q_type]['total'] += 1
        if r.get('LLMJudgeResult'):
            type_stats[q_type]['correct'] += 1

    print("\n  Per-type accuracy:")
    for q_type, stats in sorted(type_stats.items()):
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"    {q_type}: {stats['correct']}/{stats['total']} ({acc:.2f}%)")

    # Calculate per-difficulty statistics
    diff_stats = {}
    for r in judged_results:
        diff = r.get('Difficulty', 'unknown')
        if diff not in diff_stats:
            diff_stats[diff] = {'correct': 0, 'total': 0}
        diff_stats[diff]['total'] += 1
        if r.get('LLMJudgeResult'):
            diff_stats[diff]['correct'] += 1

    print("\n  Per-difficulty accuracy:")
    for diff, stats in sorted(diff_stats.items()):
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"    {diff}: {stats['correct']}/{stats['total']} ({acc:.2f}%)")

    return correct, total, type_stats, diff_stats


def run_evaluation(args):
    """Run LLM judge evaluation on inference results."""
    api_key = (args.api_key or os.environ.get('VisualCoT_GEMINI')
               or os.environ.get('VisualCoT_GEMINI') or os.environ.get('OPENAI_API_KEY'))
    if not api_key:
        print("ERROR: API key required. Use --api_key or set VisualCoT_GEMINI/VisualCoT_GEMINI environment variable.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("MMSI-Bench Evaluation")
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
        if (filename.startswith('model_results_run_') or filename.endswith('_mmsi_results.json')) and filename.endswith('.json'):
            result_files.append(filename)

    if not result_files:
        print("No result files found!")
        return 1

    print(f"Found {len(result_files)} result files: {result_files}")
    print()

    # Process each file
    all_stats = []
    all_type_stats = {}
    all_diff_stats = {}

    for filename in result_files:
        input_path = os.path.join(args.results_dir, filename)
        output_path = os.path.join(args.output_dir, f"evaluated_{filename}")

        correct, total, type_stats, diff_stats = process_results_file(
            input_path, output_path,
            api_key, args.base_url, args.judge_model,
            args.num_processes
        )
        all_stats.append({'file': filename, 'correct': correct, 'total': total})

        # Aggregate type stats
        for q_type, stats in type_stats.items():
            if q_type not in all_type_stats:
                all_type_stats[q_type] = {'correct': 0, 'total': 0}
            all_type_stats[q_type]['correct'] += stats['correct']
            all_type_stats[q_type]['total'] += stats['total']

        # Aggregate difficulty stats
        for diff, stats in diff_stats.items():
            if diff not in all_diff_stats:
                all_diff_stats[diff] = {'correct': 0, 'total': 0}
            all_diff_stats[diff]['correct'] += stats['correct']
            all_diff_stats[diff]['total'] += stats['total']

        print()

    # Print summary
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

    # Save summary
    summary = {
        'results': all_stats,
        'total_correct': total_correct,
        'total_samples': total_samples,
        'accuracy': total_correct / total_samples if total_samples > 0 else 0,
        'per_type': {k: {**v, 'accuracy': v['correct']/v['total'] if v['total'] > 0 else 0}
                     for k, v in all_type_stats.items()},
        'per_difficulty': {k: {**v, 'accuracy': v['correct']/v['total'] if v['total'] > 0 else 0}
                           for k, v in all_diff_stats.items()},
    }

    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    return 0


def run_inference(args):
    """Run model inference on MMSI-Bench."""
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("MMSI-Bench Inference with ThinkMorph")
    print("=" * 60)

    # Load model
    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(
        args.model_path, args.max_mem_per_gpu, args.vit_min_size
    )

    # Create inferencer — use DynamicVisualInterleaveInferencer for dynamic_visual_thinking
    inferencer_cls = (
        DynamicVisualInterleaveInferencer
        if args.thinking_mode == "dynamic_visual_thinking"
        else InterleaveInferencer
    )
    inferencer = inferencer_cls(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    # Thinking mode — think is always False here because we prepend our own
    # system prompt via THINKING_MODE_PROMPTS. Setting think=True would cause
    # the inferencer to add a duplicate system prompt on top of ours.
    think = False
    thinking_mode = args.thinking_mode

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
        'image_shapes': tuple(args.image_shapes) if args.image_shapes else None,
        'max_rounds': args.max_rounds,
        'force_no_visual_thinking': args.force_no_visual_thinking,
    }

    # Load data
    image_dir = os.path.join(args.output_dir, "images")
    all_data = load_mmsi_bench_data(
        data_source=args.data_source,
        parquet_path=args.parquet_path,
        image_dir=image_dir,
        cache_dir=args.cache_dir,
    )

    # Apply sharding if specified
    if args.shard is not None:
        shard_idx, num_shards = map(int, args.shard.split('/'))
        total = len(all_data)
        shard_size = (total + num_shards - 1) // num_shards
        start = shard_idx * shard_size
        end = min(start + shard_size, total)
        all_data = all_data[start:end]
        print(f"  Shard: {shard_idx}/{num_shards} (samples {start}-{end-1} of {total})")

    print(f"\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  Samples: {len(all_data)}")
    print(f"  Passes: {args.num_passes}")
    print(f"  Thinking Mode: {thinking_mode}")
    print(f"  Think: {think}")
    print()

    # Run passes
    for pass_num in range(args.num_passes):
        print(f"Starting pass {pass_num + 1}/{args.num_passes}...")

        # Determine output path up front (needed for resume)
        if args.shard is not None:
            shard_idx = int(args.shard.split('/')[0])
            output_path = os.path.join(args.output_dir, f"model_results_run_{pass_num + 1}_shard{shard_idx}.json")
        else:
            output_path = os.path.join(args.output_dir, f"model_results_run_{pass_num + 1}.json")

        # Resume: load already-processed results
        existing_results = []
        done_ids = set()
        if args.resume and os.path.exists(output_path):
            with open(output_path) as f:
                existing_results = json.load(f)
            done_ids = {r['Id'] for r in existing_results}
            print(f"  Resuming: {len(done_ids)} samples already done, {len(all_data) - len(done_ids)} remaining.")

        results = list(existing_results)
        for example in tqdm(all_data, desc=f"Pass {pass_num + 1}"):
            if example['Id'] in done_ids:
                continue
            result = process_single_sample(
                example,
                inferencer,
                inference_hyper,
                args.output_dir,
                pass_num + 1,
                think=think,
                thinking_mode=thinking_mode,
                understanding_output=getattr(args, 'understanding_output', False),
            )
            results.append(result)
            # Incrementally save after each sample so progress survives interruption
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        # Final save (also covers the no-new-samples case)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

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
        print(f"Results saved to: {output_path}")
        print(f"{'=' * 50}\n")

    print(f"\nAll passes complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"\nRun with --mode evaluate to compute accuracy scores with LLM judge.")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='MMSI-Bench Evaluation with ThinkMorph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run inference from HuggingFace dataset
    python eval_mmsi_bench.py \\
        --mode inference \\
        --model_path /path/to/ThinkMorph-7B \\
        --output_dir ./mmsi_results \\
        --num_passes 3

    # Run inference from local parquet file
    python eval_mmsi_bench.py \\
        --mode inference \\
        --model_path /path/to/ThinkMorph-7B \\
        --data_source parquet \\
        --parquet_path ./MMSI_Bench.parquet \\
        --output_dir ./mmsi_results

    # Run evaluation with LLM judge
    python eval_mmsi_bench.py \\
        --mode evaluate \\
        --results_dir ./mmsi_results \\
        --output_dir ./mmsi_evaluated \\
        --judge_model gpt-4o
"""
    )

    parser.add_argument('--mode', type=str, default='inference',
                        choices=['inference', 'evaluate'],
                        help='Mode: inference or evaluate (default: inference)')

    # Data arguments
    parser.add_argument('--data_source', type=str, default='parquet',
                        choices=['huggingface', 'parquet'],
                        help='Data source: huggingface or parquet')
    parser.add_argument('--parquet_path', type=str, default='/path/to/scratch/datasets/MMSI-Bench',
                        help='Path to parquet file (if data_source is parquet)')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='HuggingFace cache directory')

    # Model arguments
    parser.add_argument('--model_path', type=str,
                        default='/path/to/scratch/models/ThinkMorph-7B',
                        help='Path to the model directory')
    parser.add_argument('--max_mem_per_gpu', type=str, default='80GiB',
                        help='Maximum memory per GPU')
    parser.add_argument('--vit_min_size', type=int, default=512,
                        help='Minimum image size for ViT transform (default: 512)')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='/path/to/scratch/VisualCoT/mmsi_results',
                        help='Directory to save outputs')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory containing results (for evaluate mode)')

    # Inference arguments
    parser.add_argument('--num_passes', type=int, default=3,
                        help='Number of evaluation passes (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

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
    parser.add_argument('--image_shapes', type=int, nargs=2, default=None,
                        help='(H, W) shape for generated images, e.g. 720 1024')

    # Evaluation arguments
    parser.add_argument('--api_key', type=str, default=None,
                        help='Gemini API key (or set VisualCoT_GEMINI env var)')
    parser.add_argument('--base_url', type=str, default=None,
                        help='API base URL (not used for Gemini)')
    parser.add_argument('--judge_model', type=str, default='gemini-3-flash-preview',
                        help='Judge model name (default: gemini-3-flash-preview)')
    parser.add_argument('--num_processes', type=int, default=4,
                        help='Number of parallel processes for judging')
    parser.add_argument('--shard', type=str, default=None,
                        help='Shard specification as "SHARD_IDX/NUM_SHARDS" (e.g., "0/48") for parallel inference')

    # Thinking mode arguments
    parser.add_argument('--thinking_mode', type=str, default='interleaved_thinking',
                        choices=['dynamic_visual_thinking', 'visual_only_thinking',
                                 'interleaved_thinking', 'text_only_thinking', 'text_thinking', 'no_thinking', 'zebra_cot', 'understanding', 'bagel_default'],
                        help='Thinking mode (default: interleaved_thinking). Use "bagel_default" to skip the eval-side system prompt entirely and rely on BAGEL\'s own internal default prompt — only meaningful with --think.')
    parser.add_argument('--understanding_output', action='store_true',
                        help='Use understanding_output=True (ViT encoding, text-only output). Required for understanding model eval.')
    parser.add_argument('--no_think', action='store_true',
                        help='Disable thinking mode entirely')
    parser.add_argument('--max_rounds', type=int, default=3,
                        help='Max zebra_cot interleave rounds (default: 3)')
    parser.add_argument('--force_no_visual_thinking', action='store_true',
                        help='Force the inferencer to SKIP visual thinking entirely: keep the visual_only_thinking system prompt, but inject only the <image_start>/<image_end> wrapper text without generating a bridge image. Companion ablation isolating the bridge-generation forward pass.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output file, skipping already-processed samples')

    args = parser.parse_args()

    if args.mode == 'inference':
        return run_inference(args)
    else:
        if args.results_dir is None:
            args.results_dir = args.output_dir
        return run_evaluation(args)


if __name__ == '__main__':
    exit(main())
