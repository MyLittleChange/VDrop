#!/usr/bin/env python3
"""
Merge sharded BAGEL spatial inference results into a single file.

Usage:
    python merge_bagel_spatial_results.py \
        --input_dir /path/to/shard/results \
        --output_file /path/to/merged_results.json
"""

import argparse
import json
import os
import glob


def main():
    parser = argparse.ArgumentParser(
        description="Merge sharded BAGEL spatial inference results"
    )
    parser.add_argument(
        "--input_dir",
        default="/path/to/scratch/VisualCoT/training_data_mix_balance_matterport_point_matching_no_think_lora_mcqs_relative_distance_normalized",
        help="Directory containing shard result files",
    )
    parser.add_argument(
        "--output_file",
        default="/path/to/scratch/VisualCoT/training_data_mix_balance_matterport_point_matching_no_think_lora_mcqs_relative_distance_normalized/inference_results_bagel_merged.json",
        help="Output file for merged results",
    )
    parser.add_argument(
        "--pattern",
        default="inference_results_bagel*.json",
        help="Glob pattern for shard files (default: inference_results_bagel_shard*.json)",
    )

    args = parser.parse_args()

    # Find all shard files
    shard_pattern = os.path.join(args.input_dir, args.pattern)
    shard_files = sorted(glob.glob(shard_pattern))

    if not shard_files:
        print(f"No shard files found matching: {shard_pattern}")
        return 1

    print(f"Found {len(shard_files)} shard files:")
    for f in shard_files:
        print(f"  - {os.path.basename(f)}")

    # Merge results
    all_results = []
    seen_sample_ids = set()
    configs = []

    for shard_file in shard_files:
        print(f"\nLoading {os.path.basename(shard_file)}...")
        with open(shard_file, "r") as f:
            data = json.load(f)

        config = data.get("config", {})
        results = data.get("results", [])

        configs.append(config)

        # Add results, checking for duplicates
        duplicates = 0
        for result in results:
            sample_id = result.get("sample_id", "")
            if sample_id in seen_sample_ids:
                duplicates += 1
                continue
            seen_sample_ids.add(sample_id)
            all_results.append(result)

        print(f"  Loaded {len(results)} results ({duplicates} duplicates skipped)")

    # Calculate merged metrics
    total_accuracy = sum(r.get("accuracy", 0) for r in all_results)
    final_accuracy = total_accuracy / len(all_results) if all_results else 0.0

    # Calculate per-question-type accuracy
    type_stats = {}
    for r in all_results:
        q_type = r.get("question_type", "unknown")
        if q_type not in type_stats:
            type_stats[q_type] = {"correct": 0, "total": 0}
        type_stats[q_type]["total"] += 1
        type_stats[q_type]["correct"] += r.get("accuracy", 0)

    print(f"\n{'='*60}")
    print(f"Merge Complete!")
    print(f"{'='*60}")
    print(f"Total Samples: {len(all_results)}")
    print(f"Overall Accuracy: {final_accuracy:.4f} ({total_accuracy:.0f}/{len(all_results)})")

    if type_stats:
        print(f"\nPer-type accuracy:")
        for q_type, stats in sorted(type_stats.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {q_type}: {acc:.4f} ({stats['correct']:.0f}/{stats['total']})")

    # Create merged output
    merged_data = {
        "config": {
            "merged_from": [os.path.basename(f) for f in shard_files],
            "num_shards": len(shard_files),
            "model_path": configs[0].get("model_path", "") if configs else "",
            "think": configs[0].get("think", True) if configs else True,
            "num_samples": len(all_results),
        },
        "metrics": {
            "overall_accuracy": final_accuracy,
            "total_correct": total_accuracy,
            "total_samples": len(all_results),
            "per_type": {
                k: {
                    "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0,
                    "correct": v["correct"],
                    "total": v["total"],
                }
                for k, v in type_stats.items()
            },
        },
        "results": all_results,
    }

    # Save merged results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(merged_data, f, indent=2)

    print(f"\nMerged results saved to: {args.output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
