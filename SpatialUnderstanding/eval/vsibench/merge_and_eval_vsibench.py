#!/usr/bin/env python3
"""Merge sharded VSI-Bench inference results, re-extract answers with LLM, and compute metrics.

Usage:
    # Merge shards + LLM re-extraction + metrics
    python3 merge_and_eval_vsibench.py \
        --results_dir /path/to/shards \
        --run_judge \
        --judge_model gemini-3-flash-preview \
        --num_processes 4

    # Summary only from an already-combined file
    python3 merge_and_eval_vsibench.py \
        --summary /path/to/inference_results_combined.json
"""

import os
import re
import json
import argparse
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Optional


OUTPUT_FILE = "inference_results_combined.json"

NUMERIC_ANSWER_TYPES = {
    "object_counting",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
}

LETTER_ANSWER_TYPES = {
    "object_rel_distance",
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "obj_appearance_order",
    "route_planning",
}

LLM_EXTRACT_LETTER_PROMPT = """Extract the final answer letter (A, B, C, or D) from the model output below.

Question: {question}
Model output: {model_output}

Respond with ONLY a single letter: A, B, C, or D. No explanation."""

LLM_EXTRACT_NUMERIC_PROMPT = """Extract the final numeric answer from the model output below.

Question: {question}
Model output: {model_output}

Respond with ONLY a number (integer or decimal). No units, no explanation."""


def compute_mra(predicted: Optional[float], gt: float) -> float:
    """Mean Relative Accuracy averaged over thresholds {0.5, 0.55, ..., 0.95}."""
    if predicted is None or gt == 0:
        return 0.0
    rel_error = abs(predicted - gt) / abs(gt)
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    return sum(1.0 if rel_error < (1.0 - theta) else 0.0 for theta in thresholds) / 10.0


def find_shard_files(results_dir: str) -> list:
    """Find all inference_results_shard*.json files, sorted."""
    path = Path(results_dir)
    shards = sorted(path.glob("inference_results_shard*.json"))
    return [str(s) for s in shards]


def merge_shards(results_dir: str) -> list:
    """Merge all shard files into a flat list of result dicts."""
    shard_files = find_shard_files(results_dir)
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {results_dir}")

    print(f"Found {len(shard_files)} shard(s)")
    all_results = []
    for path in shard_files:
        with open(path) as f:
            data = json.load(f)
        results = data.get("results", data) if isinstance(data, dict) else data
        print(f"  {os.path.basename(path)}: {len(results)} results")
        all_results.extend(results)

    # Deduplicate by sample_id
    seen = set()
    deduped = []
    for r in all_results:
        sid = r.get("sample_id", "")
        if sid not in seen:
            seen.add(sid)
            deduped.append(r)
    if len(deduped) < len(all_results):
        print(f"  WARNING: Removed {len(all_results) - len(deduped)} duplicate sample_ids")

    print(f"  Total merged samples: {len(deduped)}")
    return deduped


def compute_metrics(results: list) -> dict:
    """Compute metrics using llm_* fields when present, falling back to original fields."""
    letter_vals = []
    numeric_vals = []

    per_type = defaultdict(lambda: {"values": [], "is_numeric": None, "count": 0})

    for r in results:
        is_numeric = r.get("is_numeric", r.get("question_type") in NUMERIC_ANSWER_TYPES)
        qt = r.get("question_type", "unknown")
        per_type[qt]["is_numeric"] = is_numeric
        per_type[qt]["count"] += 1

        if is_numeric:
            val = r.get("llm_mra", r.get("mra"))
            if val is not None:
                numeric_vals.append(val)
                per_type[qt]["values"].append(val)
        else:
            val = r.get("llm_accuracy", r.get("accuracy"))
            if val is not None:
                letter_vals.append(val)
                per_type[qt]["values"].append(val)

    per_question_type = {}
    for qt, data in per_type.items():
        mean_val = sum(data["values"]) / len(data["values"]) if data["values"] else 0.0
        metric_key = "mra" if data["is_numeric"] else "accuracy"
        per_question_type[qt] = {metric_key: mean_val, "count": data["count"]}

    return {
        "overall_accuracy_letter": sum(letter_vals) / len(letter_vals) if letter_vals else 0.0,
        "overall_mra_numeric": sum(numeric_vals) / len(numeric_vals) if numeric_vals else 0.0,
        "total_letter_samples": sum(1 for r in results if not r.get("is_numeric", r.get("question_type") in NUMERIC_ANSWER_TYPES)),
        "total_numeric_samples": sum(1 for r in results if r.get("is_numeric", r.get("question_type") in NUMERIC_ANSWER_TYPES)),
        "per_question_type": per_question_type,
    }


def print_metrics(metrics: dict):
    print("\n" + "=" * 60)
    print("VSI-Bench Evaluation Summary")
    print("=" * 60)
    print(f"  Letter samples:   {metrics['total_letter_samples']}")
    print(f"  Numeric samples:  {metrics['total_numeric_samples']}")
    print(f"  Letter Accuracy:  {metrics['overall_accuracy_letter']:.4f}")
    print(f"  Numeric MRA:      {metrics['overall_mra_numeric']:.4f}")
    print(f"\n  Per Question Type:")
    print(f"  {'Type':<40} {'Metric':>10} {'Count':>6}")
    print(f"  {'-'*40} {'-'*10} {'-'*6}")
    for qt, m in sorted(metrics["per_question_type"].items()):
        if "accuracy" in m:
            print(f"  {qt:<40} acc={m['accuracy']:.4f} {m['count']:>6}")
        else:
            print(f"  {qt:<40} mra={m['mra']:.4f} {m['count']:>6}")
    print("=" * 60)


# ============================================================================
# LLM answer re-extraction (multiprocessing, same pattern as mmsi script)
# ============================================================================

_client = None
_model_name = None


def _init_extractor_worker(api_key, model_name):
    global _client, _model_name
    if api_key:
        os.environ["VisualCoT_GEMINI"] = api_key
    from google import genai
    _client = genai.Client()
    _model_name = model_name


def _extract_single(item):
    global _client, _model_name
    NUM_RETRIES = 3

    model_output = item.get("final_answer_text", "")
    question = item.get("question", "")
    question_type = item.get("question_type", "")
    is_numeric = item.get("is_numeric", question_type in NUMERIC_ANSWER_TYPES)
    gt = item.get("ground_truth", "")

    if not model_output.strip():
        result = dict(item)
        result["llm_predicted_answer"] = None
        result["llm_accuracy"] = 0.0 if not is_numeric else None
        result["llm_mra"] = 0.0 if is_numeric else None
        return result

    prompt = (LLM_EXTRACT_NUMERIC_PROMPT if is_numeric else LLM_EXTRACT_LETTER_PROMPT).format(
        question=question,
        model_output=model_output,
    )

    for retry in range(NUM_RETRIES):
        try:
            response = _client.models.generate_content(
                model=_model_name,
                contents=prompt,
            )
            raw = str(response.text).strip()

            result = dict(item)
            result["llm_predicted_answer"] = raw

            if is_numeric:
                # Parse numeric
                nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
                predicted_num = float(nums[0]) if nums else None
                try:
                    gt_num = float(gt)
                except (ValueError, TypeError):
                    gt_num = None
                result["llm_mra"] = compute_mra(predicted_num, gt_num) if gt_num is not None else 0.0
                result["llm_accuracy"] = None
            else:
                # Parse letter
                letter_match = re.search(r"\b([A-D])\b", raw, re.IGNORECASE)
                predicted_letter = letter_match.group(1).upper() if letter_match else None
                result["llm_accuracy"] = 1.0 if predicted_letter is not None and predicted_letter == gt.strip().upper() else 0.0
                result["llm_mra"] = None

            return result

        except Exception as e:
            if retry == NUM_RETRIES - 1:
                print(f"  Error extracting sample {item.get('sample_id')}: {e}")
                result = dict(item)
                result["llm_predicted_answer"] = None
                result["llm_accuracy"] = 0.0 if not is_numeric else None
                result["llm_mra"] = 0.0 if is_numeric else None
                result["llm_extract_error"] = str(e)
                return result

    return item


def run_llm_extraction(results: list, api_key: str, model_name: str, num_processes: int = 4) -> list:
    """Re-extract answers for all results using LLM, with multiprocessing."""
    from tqdm import tqdm

    print(f"\nRunning LLM answer extraction with {model_name} ({num_processes} processes)...")
    with Pool(
        processes=num_processes,
        initializer=_init_extractor_worker,
        initargs=(api_key, model_name),
    ) as pool:
        extracted = list(tqdm(
            pool.imap(_extract_single, results),
            total=len(results),
            desc="Extracting",
        ))
    return extracted


def main():
    parser = argparse.ArgumentParser(
        description="Merge sharded VSI-Bench results, re-extract answers with LLM, and compute metrics.",
    )
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory containing inference_results_shard*.json files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save combined results (default: same as results_dir)")
    parser.add_argument("--summary", type=str, default=None,
                        help="Path to an already-combined JSON. Prints metrics and exits.")

    # LLM extraction options
    parser.add_argument("--run_judge", action="store_true",
                        help="Run LLM answer re-extraction")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Google API key (or set VisualCoT_GEMINI env var)")
    parser.add_argument("--judge_model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--num_processes", type=int, default=4)

    args = parser.parse_args()

    # ---- Summary-only mode ----
    if args.summary:
        print(f"Loading results from: {args.summary}")
        with open(args.summary) as f:
            data = json.load(f)
        results = data.get("results", data) if isinstance(data, dict) else data
        print(f"Loaded {len(results)} results")
        metrics = compute_metrics(results)
        print_metrics(metrics)
        return 0

    if not args.results_dir:
        parser.error("--results_dir is required unless --summary is used")

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Merge shards
    print("=" * 60)
    print("Merging sharded results")
    print("=" * 60)
    merged = merge_shards(args.results_dir)

    # Step 2: LLM re-extraction (optional)
    if args.run_judge:
        api_key = args.api_key or os.environ.get("VisualCoT_GEMINI")
        if not api_key:
            print("\nERROR: --run_judge requires an API key. Use --api_key or set VisualCoT_GEMINI.")
            return 1
        merged = run_llm_extraction(merged, api_key, args.judge_model, args.num_processes)

    # Step 3: Compute and print metrics
    metrics = compute_metrics(merged)
    print_metrics(metrics)

    # Step 4: Save combined results
    # Carry over config from first shard if available
    shard_files = find_shard_files(args.results_dir)
    config = {}
    if shard_files:
        with open(shard_files[0]) as f:
            first = json.load(f)
        if isinstance(first, dict) and "config" in first:
            config = dict(first["config"])
            config["shard"] = "combined"
            config["num_samples"] = len(merged)

    output = {"config": config, "metrics": metrics, "results": merged}
    out_path = os.path.join(output_dir, OUTPUT_FILE)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    return 0


if __name__ == "__main__":
    exit(main())
