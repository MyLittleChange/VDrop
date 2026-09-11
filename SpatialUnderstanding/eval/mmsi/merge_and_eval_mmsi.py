#!/usr/bin/env python
"""
Merge sharded MMSI-Bench inference results and compute evaluation metrics.

Usage:
    # Merge shards and print eval metrics
    python merge_and_eval_mmsi.py \
        --results_dir /path/to/scratch/VisualCoT/BAGEL_mmsi_eval

    # Merge shards, run LLM judge, and print eval metrics
    python merge_and_eval_mmsi.py \
        --results_dir /path/to/scratch/VisualCoT/BAGEL_mmsi_eval \
        --run_judge --judge_model gemini-3-flash-preview

    # Merge shards, run vLLM/OpenAI-compatible judge, and print eval metrics
    python merge_and_eval_mmsi.py \
        --results_dir /path/to/scratch/VisualCoT/BAGEL_mmsi_eval \
        --run_judge \
        --judge_backend vllm \
        --base_url http://localhost:8000/v1 \
        --judge_model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8

    # Specify output path for merged file
    python merge_and_eval_mmsi.py \
        --results_dir /path/to/scratch/VisualCoT/BAGEL_mmsi_eval \
        --output_dir /path/to/scratch/VisualCoT/mmsi_evaluated
"""

import os
import re
import json
import argparse
from collections import defaultdict
from multiprocessing import Pool


def find_shard_files(results_dir, run_num=1):
    """Find all shard files for a given run number."""
    shard_files = {}
    pattern = re.compile(rf"model_results_run_{run_num}_shard(\d+)\.json")
    for filename in os.listdir(results_dir):
        m = pattern.match(filename)
        if m:
            shard_idx = int(m.group(1))
            shard_files[shard_idx] = os.path.join(results_dir, filename)
    return shard_files


def merge_shards(results_dir, run_num=1, num_shards=24):
    """Merge all available shard files into a single list, sorted by Id."""
    shard_files = find_shard_files(results_dir, run_num)

    present = sorted(shard_files.keys())
    expected = set(range(num_shards))
    missing = sorted(expected - set(present))

    print(f"Run {run_num}: Found {len(present)}/{num_shards} shards")
    if missing:
        print(f"  WARNING: Missing shards: {missing}")

    all_results = []
    for shard_idx in present:
        with open(shard_files[shard_idx], "r") as f:
            data = json.load(f)
        print(f"  Shard {shard_idx}: {len(data)} samples")
        all_results.extend(data)

    # Sort by Id for consistent ordering
    all_results.sort(key=lambda x: x.get("Id", 0))

    # Check for duplicates
    ids = [r["Id"] for r in all_results]
    if len(ids) != len(set(ids)):
        dup_count = len(ids) - len(set(ids))
        print(f"  WARNING: {dup_count} duplicate IDs found. Keeping first occurrence.")
        seen = set()
        deduped = []
        for r in all_results:
            if r["Id"] not in seen:
                seen.add(r["Id"])
                deduped.append(r)
        all_results = deduped

    print(f"  Total merged samples: {len(all_results)}")
    return all_results


def compute_metrics(results):
    """Compute evaluation metrics from merged results."""
    total = len(results)
    has_answer = sum(1 for r in results if r.get("ExtractedAnswer"))
    has_error = sum(1 for r in results if r.get("Error"))
    has_generated_images = sum(1 for r in results if r.get("GeneratedImages"))

    # Per question type
    type_stats = defaultdict(lambda: {"total": 0, "has_answer": 0, "errors": 0})
    for r in results:
        qt = r.get("QuestionType", "unknown")
        type_stats[qt]["total"] += 1
        if r.get("ExtractedAnswer"):
            type_stats[qt]["has_answer"] += 1
        if r.get("Error"):
            type_stats[qt]["errors"] += 1

    # Per difficulty
    diff_stats = defaultdict(lambda: {"total": 0, "has_answer": 0, "errors": 0})
    for r in results:
        diff = r.get("Difficulty", "unknown")
        diff_stats[diff]["total"] += 1
        if r.get("ExtractedAnswer"):
            diff_stats[diff]["has_answer"] += 1
        if r.get("Error"):
            diff_stats[diff]["errors"] += 1

    # LLM judge stats (if already judged)
    judged = [r for r in results if "LLMJudgeResult" in r and r["LLMJudgeResult"] is not False]
    judge_correct = sum(1 for r in results if r.get("LLMJudgeResult") is True)

    return {
        "total": total,
        "has_answer": has_answer,
        "has_error": has_error,
        "has_generated_images": has_generated_images,
        "answer_rate": has_answer / total * 100 if total > 0 else 0,
        "judge_correct": judge_correct,
        "judge_total": len(judged),
        "per_type": dict(type_stats),
        "per_difficulty": dict(diff_stats),
    }


def print_metrics(metrics):
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 60)
    print("MMSI-Bench Evaluation Summary")
    print("=" * 60)
    print(f"  Total samples:        {metrics['total']}")
    print(f"  Has extracted answer:  {metrics['has_answer']} ({metrics['answer_rate']:.1f}%)")
    print(f"  Has errors:            {metrics['has_error']}")
    print(f"  Has generated images:  {metrics['has_generated_images']}")

    if metrics["judge_correct"] > 0 or metrics["judge_total"] > 0:
        acc = metrics["judge_correct"] / metrics["total"] * 100 if metrics["total"] > 0 else 0
        print(f"\n  LLM Judge Accuracy:    {metrics['judge_correct']}/{metrics['total']} ({acc:.2f}%)")

    print(f"\n  Per Question Type:")
    print(f"  {'Type':<40} {'Total':>6} {'Answer':>8} {'Errors':>8}")
    print(f"  {'-'*40} {'-'*6} {'-'*8} {'-'*8}")
    for qt, stats in sorted(metrics["per_type"].items()):
        ans_rate = stats["has_answer"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {qt:<40} {stats['total']:>6} {stats['has_answer']:>7} ({ans_rate:>4.0f}%) {stats['errors']:>6}")

    # Aggregated categories
    CATEGORY_MAP = {
        "Positional Relationship": "Positional Relationship",
        "Motion": "Motion",
        "Attribute": "Attribute",
        "MSR": "MSR",
    }
    cat_stats = defaultdict(lambda: {"total": 0, "has_answer": 0, "errors": 0})
    for qt, stats in metrics["per_type"].items():
        matched = False
        for prefix, cat_name in CATEGORY_MAP.items():
            if qt.startswith(prefix):
                cat_stats[cat_name]["total"] += stats["total"]
                cat_stats[cat_name]["has_answer"] += stats["has_answer"]
                cat_stats[cat_name]["errors"] += stats["errors"]
                matched = True
                break
        if not matched:
            cat_stats[qt]["total"] += stats["total"]
            cat_stats[qt]["has_answer"] += stats["has_answer"]
            cat_stats[qt]["errors"] += stats["errors"]

    print(f"\n  Aggregated Categories:")
    print(f"  {'Category':<40} {'Total':>6} {'Answer':>8} {'Errors':>8}")
    print(f"  {'-'*40} {'-'*6} {'-'*8} {'-'*8}")
    for cat, stats in sorted(cat_stats.items()):
        ans_rate = stats["has_answer"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {cat:<40} {stats['total']:>6} {stats['has_answer']:>7} ({ans_rate:>4.0f}%) {stats['errors']:>6}")

    print(f"\n  Per Difficulty:")
    print(f"  {'Difficulty':<20} {'Total':>6} {'Answer':>8} {'Errors':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8}")
    for diff, stats in sorted(metrics["per_difficulty"].items()):
        ans_rate = stats["has_answer"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {diff:<20} {stats['total']:>6} {stats['has_answer']:>7} ({ans_rate:>4.0f}%) {stats['errors']:>6}")

    print("=" * 60)


def print_judge_metrics(results):
    """Print detailed LLM judge accuracy breakdown."""
    total = len(results)
    correct = sum(1 for r in results if r.get("LLMJudgeResult") is True)
    acc = correct / total * 100 if total > 0 else 0

    print("\n" + "=" * 60)
    print("LLM Judge Evaluation Results")
    print("=" * 60)
    print(f"  Overall Accuracy: {correct}/{total} ({acc:.2f}%)")

    # Per question type
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        qt = r.get("QuestionType", "unknown")
        type_stats[qt]["total"] += 1
        if r.get("LLMJudgeResult") is True:
            type_stats[qt]["correct"] += 1

    print(f"\n  Per Question Type:")
    print(f"  {'Type':<40} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print(f"  {'-'*40} {'-'*8} {'-'*6} {'-'*10}")
    for qt, stats in sorted(type_stats.items()):
        a = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {qt:<40} {stats['correct']:>8} {stats['total']:>6} {a:>9.2f}%")

    # Aggregated categories
    CATEGORY_MAP = {
        "Positional Relationship": "Positional Relationship",
        "Motion": "Motion",
        "Attribute": "Attribute",
        "MSR": "MSR",
    }
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for qt, stats in type_stats.items():
        matched = False
        for prefix, cat_name in CATEGORY_MAP.items():
            if qt.startswith(prefix):
                cat_stats[cat_name]["correct"] += stats["correct"]
                cat_stats[cat_name]["total"] += stats["total"]
                matched = True
                break
        if not matched:
            cat_stats[qt]["correct"] += stats["correct"]
            cat_stats[qt]["total"] += stats["total"]

    print(f"\n  Aggregated Categories:")
    print(f"  {'Category':<40} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print(f"  {'-'*40} {'-'*8} {'-'*6} {'-'*10}")
    for cat, stats in sorted(cat_stats.items()):
        a = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {cat:<40} {stats['correct']:>8} {stats['total']:>6} {a:>9.2f}%")

    # Per difficulty
    diff_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        diff = r.get("Difficulty", "unknown")
        diff_stats[diff]["total"] += 1
        if r.get("LLMJudgeResult") is True:
            diff_stats[diff]["correct"] += 1

    print(f"\n  Per Difficulty:")
    print(f"  {'Difficulty':<20} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*10}")
    for diff, stats in sorted(diff_stats.items()):
        a = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {diff:<20} {stats['correct']:>8} {stats['total']:>6} {a:>9.2f}%")

    print("=" * 60)

    return {"overall_accuracy": acc, "per_type": dict(type_stats), "per_difficulty": dict(diff_stats), "per_category": dict(cat_stats)}


# ============================================================================
# LLM Judge (optional)
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
_client = None
_model_name = None
_judge_backend = None


def _init_judge_worker(api_key, model_name, judge_backend, base_url):
    global _client, _model_name, _judge_backend

    if judge_backend == "gemini":
        from google import genai
        _client = genai.Client(api_key=api_key)
    elif judge_backend == "vllm":
        from openai import OpenAI
        _client = OpenAI(
            api_key=api_key or "EMPTY",
            base_url=base_url,
            timeout=1800,
        )
    else:
        raise ValueError(f"Unsupported judge backend: {judge_backend}")

    _model_name = model_name
    _judge_backend = judge_backend


def _parse_judge_result(judge_text):
    """Parse a True/False judge response, tolerating short explanations."""
    text = re.sub(r"<think>.*?</think>", "", str(judge_text), flags=re.DOTALL | re.IGNORECASE)
    text = text.strip().lower()
    if text.startswith("true"):
        return True
    if text.startswith("false"):
        return False

    m = re.search(r"\b(true|false)\b", text)
    return m.group(1) == "true" if m else False


def _judge_single(item):
    global _client, _model_name, _judge_backend
    NUM_RETRIES = 3

    extracted = item.get("ExtractedAnswer", "")
    # Fall back to the full ModelResult when no boxed/<answer> tag was emitted
    # (e.g. when the model is run with --thinking_mode bagel_default and writes
    # the answer as free-form prose like "The answer is B").
    modeloutput = extracted if extracted else item.get("ModelResult", "")
    if not modeloutput:
        return {**item, "LLMJudgeResult": False}

    prompt = LLM_JUDGE_PROMPT.format(
        question=item.get("Question", ""),
        groundtruth=item.get("GroundTruth", ""),
        modeloutput=modeloutput,
    )

    for retry in range(NUM_RETRIES):
        try:
            if _judge_backend == "gemini":
                response = _client.models.generate_content(
                    model=_model_name,
                    contents=prompt,
                )
                judge_text = response.text
            elif _judge_backend == "vllm":
                response = _client.chat.completions.create(
                    model=_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=64,
                )
                judge_text = response.choices[0].message.content
            else:
                raise ValueError(f"Unsupported judge backend: {_judge_backend}")

            result = _parse_judge_result(judge_text)
            return {**item, "LLMJudgeResult": result}
        except Exception as e:
            if retry == NUM_RETRIES - 1:
                print(f"  Error judging ID {item.get('Id')}: {e}")
                return {**item, "LLMJudgeResult": False, "JudgeError": str(e)}
    return {**item, "LLMJudgeResult": False}


def run_llm_judge(results, api_key, model_name, num_processes=4, judge_backend="gemini", base_url=None):
    """Run LLM judge on all results using multiprocessing."""
    from tqdm import tqdm

    backend_desc = judge_backend
    if judge_backend == "vllm":
        backend_desc = f"vLLM at {base_url}"
    print(f"\nRunning LLM judge with {model_name} via {backend_desc} ({num_processes} processes)...")
    with Pool(
        processes=num_processes,
        initializer=_init_judge_worker,
        initargs=(api_key, model_name, judge_backend, base_url),
    ) as pool:
        judged = list(tqdm(
            pool.imap(_judge_single, results),
            total=len(results),
            desc="Judging",
        ))
    return judged


def _rethink_correct(r):
    """True if RethinkExtractedAnswer matches GroundTruth (MCQ exact match),
    falling back to RethinkLLMJudgeResult if the answer is not a simple letter."""
    extracted = r.get("RethinkExtractedAnswer", "")
    gt = r.get("GroundTruth", "")
    if extracted and gt:
        return extracted.strip().upper() == gt.strip().upper()
    return r.get("RethinkLLMJudgeResult") is True


def print_rethink_metrics(results):
    """Print accuracy breakdown using Rethink* fields from rethink_results_run_*.json."""
    total = len(results)
    correct = sum(1 for r in results if _rethink_correct(r))
    acc = correct / total * 100 if total > 0 else 0

    print("\n" + "=" * 60)
    print("Rethink Evaluation Results (RethinkExtractedAnswer vs GroundTruth)")
    print("=" * 60)
    print(f"  Overall Accuracy: {correct}/{total} ({acc:.2f}%)")

    # Per question type
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        qt = r.get("QuestionType", "unknown")
        type_stats[qt]["total"] += 1
        if _rethink_correct(r):
            type_stats[qt]["correct"] += 1

    print(f"\n  Per Question Type:")
    print(f"  {'Type':<40} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print(f"  {'-'*40} {'-'*8} {'-'*6} {'-'*10}")
    for qt, stats in sorted(type_stats.items()):
        a = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {qt:<40} {stats['correct']:>8} {stats['total']:>6} {a:>9.2f}%")

    # Aggregated categories
    CATEGORY_MAP = {
        "Positional Relationship": "Positional Relationship",
        "Motion": "Motion",
        "Attribute": "Attribute",
        "MSR": "MSR",
    }
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for qt, stats in type_stats.items():
        matched = False
        for prefix, cat_name in CATEGORY_MAP.items():
            if qt.startswith(prefix):
                cat_stats[cat_name]["correct"] += stats["correct"]
                cat_stats[cat_name]["total"] += stats["total"]
                matched = True
                break
        if not matched:
            cat_stats[qt]["correct"] += stats["correct"]
            cat_stats[qt]["total"] += stats["total"]

    print(f"\n  Aggregated Categories:")
    print(f"  {'Category':<40} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print(f"  {'-'*40} {'-'*8} {'-'*6} {'-'*10}")
    for cat, stats in sorted(cat_stats.items()):
        a = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {cat:<40} {stats['correct']:>8} {stats['total']:>6} {a:>9.2f}%")

    # Per difficulty
    diff_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        diff = r.get("Difficulty", "unknown")
        diff_stats[diff]["total"] += 1
        if _rethink_correct(r):
            diff_stats[diff]["correct"] += 1

    print(f"\n  Per Difficulty:")
    print(f"  {'Difficulty':<20} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*10}")
    for diff, stats in sorted(diff_stats.items()):
        a = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {diff:<20} {stats['correct']:>8} {stats['total']:>6} {a:>9.2f}%")

    print("=" * 60)

    return {"overall_accuracy": acc, "per_type": dict(type_stats), "per_difficulty": dict(diff_stats), "per_category": dict(cat_stats)}


def main():
    parser = argparse.ArgumentParser(
        description="Merge sharded MMSI-Bench results and evaluate",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/path/to/scratch/VisualCoT/BAGEL_mmsi_eval",
        help="Directory containing shard result files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save merged results (default: same as results_dir)",
    )
    parser.add_argument("--run_num", type=int, default=1, help="Run number to merge")
    parser.add_argument("--num_shards", type=int, default=24, help="Expected number of shards")

    # Summary mode: just print metrics from an already-evaluated JSON file
    parser.add_argument("--summary", type=str, default=None,
                        help="Path to an evaluated JSON file. Prints summary and exits (no merge/judge).")

    # Rethink mode: evaluate RethinkLLMJudgeResult from rethink_results_run_*.json
    parser.add_argument("--rethink", type=str, default=None,
                        help="Path to a rethink_results_run_*.json file. Prints rethink accuracy and exits.")

    # LLM Judge options
    parser.add_argument("--run_judge", action="store_true", help="Run LLM judge evaluation")
    parser.add_argument(
        "--judge_backend",
        type=str,
        default="gemini",
        choices=["gemini", "vllm"],
        help="LLM judge backend: Gemini API or OpenAI-compatible vLLM server",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Gemini API key (or VisualCoT_GEMINI/GOOGLE_API_KEY), or OpenAI-compatible key for vLLM (defaults to EMPTY for vLLM)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="OpenAI-compatible API base URL when --judge_backend=vllm",
    )
    parser.add_argument("--judge_model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--num_processes", type=int, default=4, help="Parallel processes for judging")

    args = parser.parse_args()

    # ---- Rethink mode ----
    if args.rethink:
        print(f"Loading rethink results from: {args.rethink}")
        with open(args.rethink, "r") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results")
        print_rethink_metrics(results)
        return 0

    # ---- Summary-only mode ----
    if args.summary:
        print(f"Loading evaluated results from: {args.summary}")
        with open(args.summary, "r") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results")
        print_judge_metrics(results)
        return 0

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Merge shards
    print("=" * 60)
    print("Merging sharded results")
    print("=" * 60)
    merged = merge_shards(args.results_dir, args.run_num, args.num_shards)

    # Save merged file
    merged_path = os.path.join(output_dir, f"model_results_run_{args.run_num}.json")
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged results saved to: {merged_path}")

    # Step 2: Basic metrics (no judge)
    metrics = compute_metrics(merged)
    print_metrics(metrics)

    # Step 3: Optionally run LLM judge
    if args.run_judge:
        if args.judge_backend == "gemini":
            api_key = args.api_key or os.environ.get("VisualCoT_GEMINI") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                print("\nERROR: --run_judge with --judge_backend=gemini requires --api_key, VisualCoT_GEMINI, or GOOGLE_API_KEY.")
                return 1
        else:
            api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"

        judged = run_llm_judge(
            merged,
            api_key,
            args.judge_model,
            args.num_processes,
            args.judge_backend,
            args.base_url,
        )

        # Save judged results
        judged_path = os.path.join(output_dir, f"evaluated_model_results_run_{args.run_num}.json")
        with open(judged_path, "w") as f:
            json.dump(judged, f, indent=2)
        print(f"\nJudged results saved to: {judged_path}")

        # Print judge metrics
        judge_metrics = print_judge_metrics(judged)

        # Save summary
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "total_samples": len(judged),
                    "overall_accuracy": judge_metrics["overall_accuracy"],
                    "judge_backend": args.judge_backend,
                    "judge_model": args.judge_model,
                    "judge_base_url": args.base_url if args.judge_backend == "vllm" else None,
                    "per_type": judge_metrics["per_type"],
                    "per_difficulty": judge_metrics["per_difficulty"],
                    "missing_shards": sorted(set(range(args.num_shards)) - set(find_shard_files(args.results_dir, args.run_num).keys())),
                },
                f,
                indent=2,
            )
        print(f"Summary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    exit(main())
