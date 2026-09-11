#!/usr/bin/env python3
"""
Merge sharded MindCube inference results with answer-text recovery.

Some checkpoints (notably Bagel-Zebra-CoT zero-shot) exhaust their MAX_ROUNDS
budget inside <think>...</think> without ever emitting <answer>Final Answer: X</answer>,
so the original parser stores predicted_answer=None. The reasoning text usually
still contains the verbal answer (e.g. "diagonally forward and to the left"),
which can be matched back to one of the lettered options.

This script:
  1. Loads all shards.
  2. For samples with predicted_answer=None, tries recovery strategies
     against final_answer_text in this order:
       a) "Final Answer: X" / "answer is X" pattern.
       b) <answer>...X...</answer> tag with a lettered token.
  3. Writes a new merged JSON with recovered predictions, marking each recovered
     sample with `recovery_method`, and reports both per-category accuracy and
     answer rate.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict


def recover_answer(text: str) -> tuple:
    """Return (letter, method) or (None, None)."""
    m = re.search(
        r"(?:final answer|answer is|the answer is)[:\s]+\(?([A-D])\)?",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper(), "explicit_final"

    m = re.search(r"<answer>[^<]*?\b([A-D])\b[^<]*?</answer>", text, re.IGNORECASE)
    if m:
        return m.group(1).upper(), "answer_tag"

    return None, None


def main():
    parser = argparse.ArgumentParser(description="Merge MindCube results with recovery")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--output_file", default=None)
    parser.add_argument(
        "--shard_glob",
        default="inference_results_shard{idx}.json",
        help="Per-shard filename template with {idx} placeholder",
    )
    args = parser.parse_args()

    output_file = args.output_file or os.path.join(
        args.results_dir, "merged_results_with_recovery.json"
    )

    all_results = {}
    config = {}
    for shard_idx in range(args.num_shards):
        shard_file = os.path.join(
            args.results_dir, args.shard_glob.format(idx=shard_idx)
        )
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

    method_counts = Counter()
    recovered_correct = 0
    recovered_total = 0
    for r in results:
        if r.get("predicted_answer") is not None:
            r["recovery_method"] = "original"
            continue
        text = r.get("final_answer_text", "") or ""
        letter, method = recover_answer(text)
        if letter is None:
            r["recovery_method"] = "unrecoverable"
            method_counts["unrecoverable"] += 1
            continue
        r["predicted_answer"] = letter
        gt = r.get("gt_answer")
        new_acc = 1.0 if gt is not None and letter == gt else 0.0
        r["accuracy"] = new_acc
        r["recovery_method"] = method
        method_counts[method] += 1
        recovered_total += 1
        if new_acc > 0:
            recovered_correct += 1

    answered = [r for r in results if r.get("predicted_answer") is not None]
    correct = sum(r.get("accuracy", 0) for r in results)
    overall_acc = correct / len(results) if results else 0.0
    answer_rate = len(answered) / len(results) if results else 0.0

    print("\n--- Recovery summary ---")
    for m, c in method_counts.most_common():
        print(f"  {m}: {c}")
    if recovered_total:
        print(
            f"  recovered accuracy: {recovered_correct}/{recovered_total} "
            f"= {100*recovered_correct/recovered_total:.2f}%"
        )

    print(f"\nOverall Accuracy: {overall_acc:.4f} ({correct:.0f}/{len(results)})")
    print(f"Answer Rate:      {answer_rate:.4f} ({len(answered)}/{len(results)})")

    cat0_stats = defaultdict(lambda: {"correct": 0, "total": 0, "answered": 0})
    cat1_stats = defaultdict(lambda: {"correct": 0, "total": 0, "answered": 0})
    for r in results:
        cats = r.get("category", [])
        acc = r.get("accuracy", 0)
        is_answered = r.get("predicted_answer") is not None
        if len(cats) > 0:
            cat0_stats[cats[0]]["correct"] += acc
            cat0_stats[cats[0]]["total"] += 1
            cat0_stats[cats[0]]["answered"] += int(is_answered)
        if len(cats) > 1:
            cat1_stats[cats[1]]["correct"] += acc
            cat1_stats[cats[1]]["total"] += 1
            cat1_stats[cats[1]]["answered"] += int(is_answered)

    print("\nAccuracy by category[0]:")
    cat0_acc = {}
    for cat, s in sorted(cat0_stats.items()):
        acc = s["correct"] / s["total"]
        ans = s["answered"] / s["total"]
        cat0_acc[cat] = {"accuracy": acc, "answer_rate": ans, "n": s["total"]}
        print(f"  {cat}: acc {acc:.4f} answered {ans:.4f} (n={s['total']})")

    print("\nAccuracy by category[1]:")
    cat1_acc = {}
    for cat, s in sorted(cat1_stats.items()):
        acc = s["correct"] / s["total"]
        ans = s["answered"] / s["total"]
        cat1_acc[cat] = {"accuracy": acc, "answer_rate": ans, "n": s["total"]}
        print(f"  {cat}: acc {acc:.4f} answered {ans:.4f} (n={s['total']})")

    output_data = {
        "config": config,
        "metrics": {
            "overall_accuracy": overall_acc,
            "total_correct": correct,
            "total_samples": len(results),
            "answer_rate": answer_rate,
            "recovery": {
                "method_counts": dict(method_counts),
                "recovered_total": recovered_total,
                "recovered_correct": recovered_correct,
            },
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
