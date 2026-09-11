#!/usr/bin/env python
"""
MMSI-Bench Rethink Inference Script

Re-runs inference on MMSI-Bench using pre-generated thinking images as additional
input. Tests whether the generated image's content (fed back as a 3rd input in
understanding mode) achieves the same accuracy as the original generation run.

Input per sample:
  [original_image(s)..., generated_thinking_image, question]

Output: direct answer (understanding_output=True, no new generation).

Usage:
    python eval_mmsi_bench_rethink.py \
        --results_dir /path/to/prior/mmsi/results \
        --model_path /path/to/model \
        --output_dir /path/to/rethink/output
"""

import os
import json
import argparse
import regex
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
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

# ============================================================================
# LLM Judge (same as eval_mmsi_bench.py)
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

client_judge = None
JUDGE_MODEL_NAME_GLOBAL = None


def init_worker(api_key, base_url, model_name):
    global client_judge, JUDGE_MODEL_NAME_GLOBAL
    from google import genai
    if api_key:
        os.environ['VisualCoT_GEMINI'] = api_key
    client_judge = genai.Client()
    JUDGE_MODEL_NAME_GLOBAL = model_name


def judge_single_result(item):
    global client_judge, JUDGE_MODEL_NAME_GLOBAL
    NUM_RETRIES = 3
    for retry in range(NUM_RETRIES):
        try:
            question = item.get('Question', '')
            groundtruth = item.get('GroundTruth', '')
            extracted = item.get('ExtractedAnswer', '')
            if not extracted:
                return {**item, 'LLMJudgeResult': False}
            response = client_judge.models.generate_content(
                model=JUDGE_MODEL_NAME_GLOBAL,
                contents=LLM_JUDGE_PROMPT.format(
                    question=question, groundtruth=groundtruth, modeloutput=extracted
                )
            )
            judge_response = str(response.text).strip().lower()
            return {**item, 'LLMJudgeResult': 'true' in judge_response}
        except Exception as e:
            if retry == NUM_RETRIES - 1:
                print(f"Error judging ID {item.get('Id')}: {e}")
                return {**item, 'LLMJudgeResult': False, 'JudgeError': str(e)}
    return {**item, 'LLMJudgeResult': False}


# ============================================================================
# Model Loading
# ============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    tokenizer, new_token_ids, _ = add_dynamic_visual_tokens(tokenizer, new_token_ids)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, vit_min_size, 14)

    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        'language_model.model.embed_tokens', 'time_embedder', 'latent_pos_embed',
        'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed'
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
        model, checkpoint=checkpoint_path, device_map=device_map,
        offload_buffers=True, dtype=torch.bfloat16, force_hooks=True,
        offload_folder="/tmp/offload"
    )
    model = model.eval()
    print('Model loaded successfully')

    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


# ============================================================================
# Answer Extraction
# ============================================================================

def extract_boxed_answer(text):
    if text is None:
        return None
    # 1. \boxed{}
    pattern = r'\\boxed\{((?:[^{}]|{(?:[^{}]|{.*})*})*)\}'
    matches = regex.findall(pattern, text)
    if matches:
        return matches[-1]
    # 2. <answer> tag — also normalise "B: text" inside tag to just "B"
    pattern_alt = r'<answer>(.*?)</answer>'
    matches_alt = regex.findall(pattern_alt, text, regex.DOTALL)
    if matches_alt:
        ans = matches_alt[-1].strip()
        lm = regex.match(r'^([A-Da-d])[:\.\s]', ans)
        if lm:
            return lm.group(1).upper()
        return ans
    # 3. Leading "X:" / "X." / "X " / "X\n" at start of output (BAGEL baseline format)
    m = regex.match(r'^([A-Da-d])[:\.\s\n]', text.strip())
    if m:
        return m.group(1).upper()
    # 4. "the answer is X" / "correct answer is X"
    m = regex.search(r'(?:answer is|correct answer is)(?:\s+option)?\s+([A-Da-d])\b', text, regex.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 5. Standalone letter at end of text
    m = regex.search(r'\b([A-Da-d])\s*\.?\s*$', text.strip())
    if m:
        return m.group(1).upper()
    return None


# ============================================================================
# Rethink Inference
# ============================================================================

def process_single_rethink_sample(
    prior_result: Dict[str, Any],
    inferencer: InterleaveInferencer,
    inference_hyper: Dict[str, Any],
    thinking_mode: str = "no_thinking",
) -> Dict[str, Any]:
    """Re-run inference feeding original images + generated thinking image as input."""
    NUM_RETRIES = 3

    for retry in range(NUM_RETRIES):
        try:
            task_id = prior_result['Id']
            question = prior_result['Question']
            image_paths = prior_result.get('ImagePaths', [])
            generated_images = prior_result.get('GeneratedImages', [])

            if not generated_images:
                return {**prior_result,
                        'RethinkModelResult': '',
                        'RethinkExtractedAnswer': '',
                        'RethinkLLMJudgeResult': False,
                        'Error': 'No generated images in prior result'}

            # Load original image(s) + generated thinking image
            input_list = []
            for img_path in image_paths:
                if os.path.exists(img_path):
                    input_list.append(Image.open(img_path).convert('RGB'))
                else:
                    print(f"Warning: image not found: {img_path}")

            generated_image_path = generated_images[0]
            if not os.path.exists(generated_image_path):
                return {**prior_result,
                        'RethinkModelResult': '',
                        'RethinkExtractedAnswer': '',
                        'RethinkLLMJudgeResult': False,
                        'Error': f'Generated image not found: {generated_image_path}'}

            input_list.append(Image.open(generated_image_path).convert('RGB'))

            # Build prompt (strip the \boxed instruction, use our thinking mode prompt instead)
            system_prompt = THINKING_MODE_PROMPTS[thinking_mode]
            original_question = prior_result.get('OriginalQuestion', question)
            full_text = system_prompt + "\n" + original_question + \
                "\nThink about the question and give your final answer in \\boxed{Answer} format."
            input_list.append(full_text)

            # Run understanding-only inference (no new generation)
            output_list = inferencer(
                input_list=input_list,
                understanding_output=True,
                think=False,
                **inference_hyper
            )

            model_output_result = ""
            for out_item in output_list:
                if isinstance(out_item, str):
                    model_output_result += out_item

            model_output_result = model_output_result.strip()
            extracted_answer = extract_boxed_answer(model_output_result)

            return {
                **prior_result,
                'RethinkModelResult': model_output_result,
                'RethinkExtractedAnswer': extracted_answer if extracted_answer else "",
                'RethinkLLMJudgeResult': False,
                'RethinkGeneratedImagePath': generated_image_path,
            }

        except Exception as e:
            if retry < NUM_RETRIES - 1:
                print(f"Error processing ID {prior_result.get('Id')}: {e}. Retrying ({retry+1}/{NUM_RETRIES})...")
            else:
                print(f"Error processing ID {prior_result.get('Id')}: {e}. All retries failed.")

    return {
        **prior_result,
        'RethinkModelResult': '',
        'RethinkExtractedAnswer': '',
        'RethinkLLMJudgeResult': False,
        'Error': str(e),
    }


# ============================================================================
# Evaluation
# ============================================================================

def run_evaluation(args):
    """Run LLM judge on rethink results."""
    from google import genai
    api_key = args.api_key or os.environ.get('VisualCoT_GEMINI')
    if not api_key:
        print("ERROR: API key required. Use --api_key or set VisualCoT_GEMINI.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    result_files = sorted(
        f for f in os.listdir(args.results_dir)
        if f.startswith('rethink_results_run_') and f.endswith('.json')
    )
    if not result_files:
        print("No rethink result files found!")
        return 1

    print(f"Found {len(result_files)} result files")
    all_stats = []
    all_type_stats = {}

    for filename in result_files:
        input_path = os.path.join(args.results_dir, filename)
        output_path = os.path.join(args.output_dir, f"evaluated_{filename}")

        with open(input_path, 'r') as f:
            results = json.load(f)

        # Prepare items for judging (use rethink fields)
        items_to_judge = [
            {**r, 'ExtractedAnswer': r.get('RethinkExtractedAnswer', '')}
            for r in results
        ]

        with Pool(
            processes=args.num_processes,
            initializer=init_worker,
            initargs=(api_key, None, args.judge_model)
        ) as pool:
            judged = list(tqdm(pool.imap(judge_single_result, items_to_judge),
                               total=len(items_to_judge), desc=f"Judging {filename}"))

        # Merge RethinkLLMJudgeResult back
        for orig, j in zip(results, judged):
            orig['RethinkLLMJudgeResult'] = j.get('LLMJudgeResult', False)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        correct = sum(1 for r in results if r.get('RethinkLLMJudgeResult'))
        orig_correct = sum(1 for r in results if r.get('LLMJudgeResult'))
        total = len(results)
        print(f"{filename}: rethink={correct}/{total} ({correct/total*100:.2f}%) | "
              f"original={orig_correct}/{total} ({orig_correct/total*100:.2f}%)")

        all_stats.append({'file': filename, 'correct': correct, 'total': total,
                          'orig_correct': orig_correct})

        for r in results:
            q_type = r.get('QuestionType', 'unknown')
            if q_type not in all_type_stats:
                all_type_stats[q_type] = {'correct': 0, 'orig_correct': 0, 'total': 0}
            all_type_stats[q_type]['total'] += 1
            if r.get('RethinkLLMJudgeResult'):
                all_type_stats[q_type]['correct'] += 1
            if r.get('LLMJudgeResult'):
                all_type_stats[q_type]['orig_correct'] += 1

    total_correct = sum(s['correct'] for s in all_stats)
    total_orig = sum(s['orig_correct'] for s in all_stats)
    total_samples = sum(s['total'] for s in all_stats)

    print(f"\n{'='*60}")
    print(f"Overall rethink: {total_correct}/{total_samples} ({total_correct/total_samples*100:.2f}%)")
    print(f"Overall original: {total_orig}/{total_samples} ({total_orig/total_samples*100:.2f}%)")
    print(f"\nPer-type:")
    for q_type, stats in sorted(all_type_stats.items()):
        t = stats['total']
        print(f"  {q_type}: rethink={stats['correct']}/{t} ({stats['correct']/t*100:.2f}%) | "
              f"orig={stats['orig_correct']}/{t} ({stats['orig_correct']/t*100:.2f}%)")

    summary = {
        'rethink_accuracy': total_correct / total_samples if total_samples > 0 else 0,
        'original_accuracy': total_orig / total_samples if total_samples > 0 else 0,
        'total_correct': total_correct,
        'total_samples': total_samples,
        'per_type': {k: {**v, 'rethink_accuracy': v['correct']/v['total'] if v['total'] > 0 else 0,
                         'original_accuracy': v['orig_correct']/v['total'] if v['total'] > 0 else 0}
                     for k, v in all_type_stats.items()},
    }
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    return 0


# ============================================================================
# Inference Entry Point
# ============================================================================

def run_inference(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("MMSI-Bench Rethink Inference")
    print("=" * 60)

    # Load all prior shard results from results_dir
    shard_files = sorted(
        f for f in os.listdir(args.results_dir)
        if f.startswith('model_results_run_1') and f.endswith('.json')
        and 'evaluated' not in f
    )
    if not shard_files:
        print(f"ERROR: No model_results_run_1*.json files found in {args.results_dir}")
        return 1

    print(f"Loading prior results from: {shard_files}")
    all_data = []
    for fname in shard_files:
        with open(os.path.join(args.results_dir, fname), 'r') as f:
            all_data.extend(json.load(f))

    # Keep only samples with generated images
    all_data = [r for r in all_data if r.get('GeneratedImages')]
    print(f"Samples with generated images: {len(all_data)}")

    # Apply sharding if specified
    if args.shard is not None:
        shard_idx, num_shards = map(int, args.shard.split('/'))
        total = len(all_data)
        shard_size = (total + num_shards - 1) // num_shards
        start = shard_idx * shard_size
        end = min(start + shard_size, total)
        all_data = all_data[start:end]
        print(f"Shard {shard_idx}/{num_shards}: samples {start}-{end-1} ({len(all_data)} samples)")

    # Load checkpoint if resuming
    if args.shard is not None:
        shard_idx = int(args.shard.split('/')[0])
        output_path = os.path.join(args.output_dir, f"rethink_results_run_1_shard{shard_idx}.json")
    else:
        output_path = os.path.join(args.output_dir, "rethink_results_run_1.json")

    completed_ids = set()
    completed_results = {}
    if not args.no_resume and os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing = json.load(f)
        completed_results = {r['Id']: r for r in existing if r.get('RethinkModelResult', '').strip()}
        completed_ids = set(completed_results.keys())
        print(f"Resuming: {len(completed_ids)} already done")

    # Load model
    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(
        args.model_path, args.max_mem_per_gpu, args.vit_min_size
    )
    inferencer = InterleaveInferencer(
        model=model, vae_model=vae_model, tokenizer=tokenizer,
        vae_transform=vae_transform, vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

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
    }

    print(f"\nConfiguration:")
    print(f"  Model:         {args.model_path}")
    print(f"  Results dir:   {args.results_dir}")
    print(f"  Thinking mode: {args.thinking_mode}")
    print(f"  Samples:       {len(all_data)}")
    print()

    results = dict(completed_results)
    samples_to_process = [r for r in all_data if r['Id'] not in completed_ids]

    for i, prior_result in enumerate(tqdm(samples_to_process, desc="Rethink inference")):
        result = process_single_rethink_sample(
            prior_result, inferencer, inference_hyper, args.thinking_mode
        )
        results[result['Id']] = result

        # Save checkpoint every 10 samples
        if (i + 1) % 10 == 0:
            with open(output_path, 'w') as f:
                json.dump(list(results.values()), f, indent=2)

    final_results = list(results.values())
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    has_answer = sum(1 for r in final_results if r.get('RethinkExtractedAnswer'))
    print(f"\n{'='*50}")
    print(f"Rethink Inference Complete")
    print(f"Total: {len(final_results)}")
    print(f"Has Extracted Answer: {has_answer}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*50}")
    return 0


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MMSI-Bench Rethink Inference — feed pre-generated images back as input'
    )
    parser.add_argument('--mode', choices=['inference', 'evaluate'], default='inference')

    # Inference args
    parser.add_argument('--results_dir', required=True,
                        help='Directory containing prior model_results_run_1*.json files with GeneratedImages')
    parser.add_argument('--model_path',
                        default='/path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_panorama_qa_visual_only')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_mem_per_gpu', default='80GiB')
    parser.add_argument('--vit_min_size', type=int, default=512)
    parser.add_argument('--thinking_mode', choices=['no_thinking', 'text_only_thinking', 'understanding'],
                        default='no_thinking')
    parser.add_argument('--shard', type=str, default=None,
                        help="Shard in format 'index/total' (e.g., '0/4')")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_resume', action='store_true')

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
    parser.add_argument('--image_shapes', type=int, nargs=2, default=None)

    # Evaluation args
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--judge_model', type=str, default='gemini-3-flash-preview')
    parser.add_argument('--num_processes', type=int, default=4)

    args = parser.parse_args()

    if args.mode == 'inference':
        return run_inference(args)
    else:
        return run_evaluation(args)


if __name__ == '__main__':
    main()
