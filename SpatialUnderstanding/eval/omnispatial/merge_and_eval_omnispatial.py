#!/usr/bin/env python3
"""
Merge sharded BAGEL OmniSpatial inference results and compute global metrics.

Usage:
    python merge_and_eval_omnispatial.py \
        --results_dir /path/to/scratch/omnispatial/BAGEL_format_topdown_qa_visual_only

    # Save merged output to a custom path:
    python merge_and_eval_omnispatial.py \
        --results_dir /path/to/shards \
        --output_file /path/to/merged.json

    # Print only, no file written:
    python merge_and_eval_omnispatial.py \
        --results_dir /path/to/shards \
        --print_only
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Merge sharded BAGEL OmniSpatial results and compute global metrics"
    )
    parser.add_argument(
        "--results_dir",
        default="/path/to/scratch/VisualCoT/omnispatial/training_data_mix_balance_matterport_point_matching_no_think_lora",
        # required=True,
        help="Directory containing shard result files (inference_results_bagel_shard*.json)",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output file for merged results (default: <results_dir>/inference_results_bagel_merged.json)",
    )
    parser.add_argument(
        "--pattern",
        default="inference_results_bagel_shard*.json",
        help="Glob pattern for shard files",
    )
    parser.add_argument(
        "--print_only",
        action="store_true",
        help="Print metrics without writing output file",
    )
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = os.path.join(
            args.results_dir, "inference_results_bagel_merged.json"
        )

    # ── Find shard files ──────────────────────────────────────────────────────
    shard_pattern = os.path.join(args.results_dir, args.pattern)
    shard_files = sorted(glob.glob(shard_pattern))
    shard_files = [f for f in shard_files if "checkpoint" not in os.path.basename(f) and "merged" not in os.path.basename(f)]

    if not shard_files:
        print(f"ERROR: No shard files found matching: {shard_pattern}")
        return 1

    print(f"Found {len(shard_files)} shard file(s):")
    for f in shard_files:
        print(f"  - {os.path.basename(f)}")

    # ── Merge results ─────────────────────────────────────────────────────────
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
            sample_id = r.get("_idx", r.get("id", ""))
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

    # ── Recompute global metrics ──────────────────────────────────────────────
    total_correct = sum(r.get("accuracy", 0) for r in all_results)
    overall_accuracy = total_correct / len(all_results)

    # Per task_type → per sub_task_type
    task_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    for r in all_results:
        task = r.get("task_type", "unknown")
        sub = r.get("sub_task_type", "unknown")
        acc = r.get("accuracy", 0)
        task_stats[task]["_overall"]["correct"] += acc
        task_stats[task]["_overall"]["total"] += 1
        task_stats[task][sub]["correct"] += acc
        task_stats[task][sub]["total"] += 1

    by_task_type = {}
    for task, subs in sorted(task_stats.items()):
        entry = {}
        overall_s = subs["_overall"]
        entry["overall"] = overall_s["correct"] / overall_s["total"]
        entry["n"] = overall_s["total"]
        for sub, s in sorted(subs.items()):
            if sub == "_overall":
                continue
            entry[sub] = s["correct"] / s["total"]
        by_task_type[task] = entry

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"OmniSpatial — Merged Results")
    print(f"{'='*60}")
    print(f"Total samples : {len(all_results)}")
    print(f"Overall acc   : {overall_accuracy:.4f}  ({total_correct:.0f}/{len(all_results)})")
    print()
    print(f"{'Task Type':<30} {'Acc':>6}  {'N':>5}")
    print(f"{'-'*45}")
    for task, entry in sorted(by_task_type.items()):
        print(f"  {task:<28} {entry['overall']:.4f}  {entry['n']:>5}")
        for k, v in sorted(entry.items()):
            if k in ("overall", "n"):
                continue
            n = task_stats[task][k]["total"]
            print(f"    {k:<26} {v:.4f}  {n:>5}")
    print(f"{'='*60}")

    # ── Tab-separated row for spreadsheet copy-paste ─────────────────────────
    category_order = ["Complex_Logic", "Dynamic_Reasoning", "Perspective_Taking", "Spatial_Interaction"]
    header = ["Overall"] + category_order
    values = [f"{overall_accuracy:.4f}"]
    for cat in category_order:
        if cat in by_task_type:
            values.append(f"{by_task_type[cat]['overall']:.4f}")
        else:
            values.append("N/A")
    print(f"\n--- Copy-paste row (one value per line, paste into first cell of the row) ---")
    for h, v in zip(header, values):
        print(f"{h}: {v}")
    print()
    print("Values only (select all, paste into Google Sheets cell A1, then Data > Split text to columns):")
    print("  ".join(values))

    if args.print_only:
        return 0

    # ── Write merged output ───────────────────────────────────────────────────
    first_cfg = configs[0] if configs else {}
    merged = {
        "config": {
            "merged_from": [os.path.basename(f) for f in shard_files],
            "num_shards": len(shard_files),
            "model_path": first_cfg.get("model_path", ""),
            "think": first_cfg.get("think", True),
            "thinking_mode": first_cfg.get("thinking_mode", ""),
            "num_samples": len(all_results),
        },
        "metrics": {
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_samples": len(all_results),
            "by_task_type": by_task_type,
        },
        "results": all_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged results saved to: {args.output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
