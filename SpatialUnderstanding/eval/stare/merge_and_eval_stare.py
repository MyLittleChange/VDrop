#!/usr/bin/env python3
"""
Merge sharded BAGEL STARE inference results and compute global metrics.

Usage:
    python merge_and_eval_stare.py \
        --results_dir /path/to/scratch/VisualCoT/stare_perspective/BAGEL_format_training_data_mix_all_no_thinking

    python merge_and_eval_stare.py --results_dir <dir> --print_only
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Merge sharded BAGEL STARE results and compute global metrics"
    )
    parser.add_argument("--results_dir", default='/path/to/scratch/VisualCoT/stare_perspective/BAGEL_format_training_data_mix_all_rotation_balance_visual_only',
                        help="Directory containing shard result files (inference_results_bagel_shard*.json)")
    parser.add_argument("--output_file", default=None,
                        help="Output file for merged results (default: <results_dir>/inference_results_bagel_merged.json)")
    parser.add_argument("--pattern", default="inference_results_bagel_shard*.json",
                        help="Glob pattern for shard files")
    parser.add_argument("--print_only", action="store_true",
                        help="Print metrics without writing output file")
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = os.path.join(args.results_dir, "inference_results_bagel_merged.json")

    shard_pattern = os.path.join(args.results_dir, args.pattern)
    shard_files = sorted(glob.glob(shard_pattern))
    shard_files = [f for f in shard_files if "checkpoint" not in os.path.basename(f) and "merged" not in os.path.basename(f)]

    if not shard_files:
        print(f"ERROR: No shard files found matching: {shard_pattern}")
        return 1

    print(f"Found {len(shard_files)} shard file(s):")
    for f in shard_files:
        print(f"  - {os.path.basename(f)}")

    all_results = []
    seen_ids = set()
    configs = []

    for shard_file in shard_files:
        with open(shard_file) as f:
            data = json.load(f)

        configs.append(data.get("config", {}))
        results = data.get("results", [])

        duplicates = 0
        for r in results:
            sample_id = r.get("_idx", r.get("qid", ""))
            if sample_id in seen_ids:
                duplicates += 1
                continue
            seen_ids.add(sample_id)
            all_results.append(r)

        status = f"{len(results)} results"
        if duplicates:
            status += f", {duplicates} duplicates skipped"
        print(f"  {os.path.basename(shard_file)}: {status}")

    if not all_results:
        print("ERROR: No results found after merging.")
        return 1

    total_correct = sum(r.get("accuracy", 0) for r in all_results)
    overall_accuracy = total_correct / len(all_results)

    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    choice_stats = defaultdict(int)
    for r in all_results:
        cat = r.get("category", "unknown")
        cat_stats[cat]["correct"] += r.get("accuracy", 0)
        cat_stats[cat]["total"] += 1
        choice_stats[str(r.get("predicted_answer", "?"))] += 1

    print(f"\n{'='*60}")
    print(f"STARE — Merged Results")
    print(f"{'='*60}")
    print(f"Total samples : {len(all_results)}")
    print(f"Overall acc   : {overall_accuracy:.4f}  ({total_correct:.0f}/{len(all_results)})")

    if len(cat_stats) > 1:
        print()
        print(f"{'Category':<30} {'Acc':>6}  {'N':>5}")
        print(f"{'-'*45}")
        for cat in sorted(cat_stats):
            s = cat_stats[cat]
            print(f"  {cat:<28} {s['correct']/s['total']:.4f}  {s['total']:>5}")

    print()
    print("Predicted answer distribution:")
    for k in sorted(choice_stats):
        print(f"  {k}: {choice_stats[k]}")
    print(f"{'='*60}")

    print(f"\n--- Copy-paste row ---")
    print(f"Overall: {overall_accuracy:.4f}")
    print(f"N: {len(all_results)}")

    if args.print_only:
        return 0

    first_cfg = configs[0] if configs else {}
    merged = {
        "config": {
            "merged_from": [os.path.basename(f) for f in shard_files],
            "num_shards": len(shard_files),
            "model_path": first_cfg.get("model_path", ""),
            "think": first_cfg.get("think", False),
            "thinking_mode": first_cfg.get("thinking_mode", ""),
            "num_samples": len(all_results),
        },
        "metrics": {
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_samples": len(all_results),
            "by_category": {k: v["correct"] / v["total"] for k, v in cat_stats.items()},
        },
        "results": all_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged results saved to: {args.output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
