#!/usr/bin/env python3
"""
Merge sharded BLINK Multi-view_Reasoning inference results and compute final accuracy.

Usage:
    python3 merge_blink_results.py --results_dir /path/to/dir --num_shards 2
    python3 merge_blink_results.py --results_dir /path/to/dir  # auto-detects shards
"""

import argparse
import json
import os
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Merge sharded BLINK inference results")
    parser.add_argument("--results_dir",default='/path/to/scratch/blink/training_data_mix_balance_matterport_point_matching_no_think_lora', help="Directory containing shard JSON files")
    parser.add_argument("--num_shards", type=int, default=None,
                        help="Number of shards (auto-detected if not specified)")
    parser.add_argument("--output_file", default=None,
                        help="Output merged JSON (default: results_dir/merged_results.json)")
    args = parser.parse_args()

    output_file = args.output_file or os.path.join(args.results_dir, "merged_results.json")

    # Auto-detect shards if not specified
    if args.num_shards is None:
        shard_files = sorted(
            f for f in os.listdir(args.results_dir)
            if f.startswith("inference_results_shard") and f.endswith(".json")
        )
        num_shards = len(shard_files)
        print(f"Auto-detected {num_shards} shard(s): {shard_files}")
    else:
        num_shards = args.num_shards

    # Load all shards
    all_results = {}
    config = {}
    for shard_idx in range(num_shards):
        shard_file = os.path.join(args.results_dir, f"inference_results_shard{shard_idx}.json")
        if not os.path.exists(shard_file):
            print(f"WARNING: Missing shard file: {shard_file}")
            continue
        with open(shard_file) as f:
            data = json.load(f)
        shard_results = data.get("results", [])
        if not config:
            config = data.get("config", {})
        for r in shard_results:
            sid = r.get("sample_id", "")
            if sid and sid not in all_results:
                all_results[sid] = r
        print(f"Shard {shard_idx}: {len(shard_results)} samples")

    results = list(all_results.values())
    print(f"\nTotal unique samples: {len(results)}")

    if not results:
        print("No results found.")
        return

    # Overall accuracy
    answered = [r for r in results if r.get("predicted_answer") is not None]
    correct = sum(r.get("accuracy", 0) for r in results)
    overall_acc = correct / len(results)
    answer_rate = len(answered) / len(results)

    print(f"Overall Accuracy: {overall_acc:.4f} ({correct:.0f}/{len(results)})")
    print(f"Answer Rate:      {answer_rate:.4f} ({len(answered)}/{len(results)})")

    # Per sub_task breakdown
    subtask_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        st = r.get("sub_task", "unknown")
        subtask_stats[st]["correct"] += r.get("accuracy", 0)
        subtask_stats[st]["total"] += 1

    print("\nAccuracy by sub_task:")
    subtask_acc = {}
    for st, s in sorted(subtask_stats.items()):
        acc = s["correct"] / s["total"]
        subtask_acc[st] = acc
        print(f"  {st}: {acc:.4f} (n={s['total']})")

    # Save merged output
    output_data = {
        "config": config,
        "metrics": {
            "overall_accuracy": overall_acc,
            "total_correct": correct,
            "total_samples": len(results),
            "answer_rate": answer_rate,
            "by_sub_task": subtask_acc,
            "sub_task_counts": {st: s["total"] for st, s in subtask_stats.items()},
        },
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nMerged results saved to: {output_file}")


if __name__ == "__main__":
    main()
