#!/usr/bin/env python3
"""
Merge sharded MindCube inference results and compute final accuracy breakdown.
"""

import argparse
import json
import os
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Merge sharded MindCube results")
    parser.add_argument("--results_dir", required=True, help="Directory containing shard JSON files")
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--output_file", default=None, help="Output merged JSON (default: results_dir/merged_results.json)")
    args = parser.parse_args()

    output_file = args.output_file or os.path.join(args.results_dir, "merged_results.json")

    # Load all shards
    all_results = {}
    config = {}
    for shard_idx in range(args.num_shards):
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

    # Overall accuracy
    answered = [r for r in results if r.get("predicted_answer") is not None]
    correct = sum(r.get("accuracy", 0) for r in results)
    overall_acc = correct / len(results) if results else 0.0
    answer_rate = len(answered) / len(results) if results else 0.0

    print(f"Overall Accuracy: {overall_acc:.4f} ({correct:.0f}/{len(results)})")
    print(f"Answer Rate:      {answer_rate:.4f} ({len(answered)}/{len(results)})")

    # Per category[0] breakdown
    cat0_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    cat1_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cats = r.get("category", [])
        acc = r.get("accuracy", 0)
        if len(cats) > 0:
            cat0_stats[cats[0]]["correct"] += acc
            cat0_stats[cats[0]]["total"] += 1
        if len(cats) > 1:
            cat1_stats[cats[1]]["correct"] += acc
            cat1_stats[cats[1]]["total"] += 1

    print("\nAccuracy by category[0]:")
    cat0_acc = {}
    for cat, s in sorted(cat0_stats.items()):
        acc = s["correct"] / s["total"]
        cat0_acc[cat] = acc
        print(f"  {cat}: {acc:.4f} (n={s['total']})")

    print("\nAccuracy by category[1]:")
    cat1_acc = {}
    for cat, s in sorted(cat1_stats.items()):
        acc = s["correct"] / s["total"]
        cat1_acc[cat] = acc
        print(f"  {cat}: {acc:.4f} (n={s['total']})")

    # Save merged output
    output_data = {
        "config": config,
        "metrics": {
            "overall_accuracy": overall_acc,
            "total_correct": correct,
            "total_samples": len(results),
            "answer_rate": answer_rate,
            "by_category_0": cat0_acc,
            "by_category_1": cat1_acc,
        },
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nMerged results saved to: {output_file}")


if __name__ == "__main__":
    main()
