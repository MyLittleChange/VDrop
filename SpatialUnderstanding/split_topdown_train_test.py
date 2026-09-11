"""
Split Gemini topdown results into train/test sets (90/10) by scene_id.
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict


def compute_metrics(results):
    per_type = defaultdict(lambda: {"correct": 0.0, "total": 0})
    total_correct = 0.0
    for item in results:
        qtype = item["question_type"]
        per_type[qtype]["correct"] += item["accuracy"]
        per_type[qtype]["total"] += 1
        total_correct += item["accuracy"]
    total = len(results)
    metrics = {
        "overall_accuracy": round(total_correct / total, 4) if total > 0 else 0.0,
        "total_correct": total_correct,
        "total_samples": total,
        "per_type": {},
    }
    for qtype, vals in per_type.items():
        metrics["per_type"][qtype] = {
            "accuracy": round(vals["correct"] / vals["total"], 4) if vals["total"] > 0 else 0.0,
            "correct": vals["correct"],
            "total": vals["total"],
        }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Split Gemini topdown results into train/test by scene_id (90/10)")
    parser.add_argument(
        "--input",
        default="/path/to/scratch/VisualCoT/Gemini_topdown/gemini_3_pro_preview_topdown_high.json",
        help="Path to the Gemini topdown JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (defaults to same directory as --input)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of scene_ids for train (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Load data
    with open(args.input) as f:
        data = json.load(f)

    results = data["results"]
    print(f"Total samples: {len(results)}")

    # Group samples by scene_id
    scene_to_samples = defaultdict(list)
    for item in results:
        scene_to_samples[item["scene_id"]].append(item)

    scene_ids = sorted(scene_to_samples.keys())
    print(f"Unique scene_ids: {len(scene_ids)}")

    # Shuffle and split scene_ids
    random.seed(args.seed)
    random.shuffle(scene_ids)
    n_train = int(len(scene_ids) * args.train_ratio)
    train_scenes = set(scene_ids[:n_train])
    test_scenes = set(scene_ids[n_train:])

    # Split samples
    train_results = []
    test_results = []
    for scene_id, samples in scene_to_samples.items():
        if scene_id in train_scenes:
            train_results.extend(samples)
        else:
            test_results.extend(samples)

    # Compute metrics
    train_metrics = compute_metrics(train_results)
    test_metrics = compute_metrics(test_results)

    print(f"Train: {len(train_results)} samples, {len(train_scenes)} scenes, accuracy: {train_metrics['overall_accuracy']}")
    print(f"Test:  {len(test_results)} samples, {len(test_scenes)} scenes, accuracy: {test_metrics['overall_accuracy']}")

    # Save
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, split_results, metrics in [("train", train_results, train_metrics), ("test", test_results, test_metrics)]:
        split_data = {
            "config": data.get("config", {}),
            "metrics": metrics,
            "split": split,
            "num_samples": len(split_results),
            "results": split_results,
        }
        out_path = output_dir / f"gemini_3_pro_preview_topdown_high_{split}.json"
        with open(out_path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {split} to {out_path}")


if __name__ == "__main__":
    main()
