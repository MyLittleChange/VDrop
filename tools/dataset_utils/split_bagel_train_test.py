"""
Split BAGEL inference results into train/test sets.

Training set: sample_ids that appear in the Gemini topdown file.
Test set: all other sample_ids.
"""

import json
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Split BAGEL results into train/test based on Gemini topdown sample_ids")
    parser.add_argument(
        "--bagel_file",
        type=str,
        default="/path/to/scratch/VisualCoT/spatial_topdown_visual_only_Spatial_nomalized_all_results/inference_results_bagel_merged.json",
        help="Path to BAGEL merged inference results",
    )
    parser.add_argument(
        "--gemini_file",
        type=str,
        default="/path/to/scratch/VisualCoT/Gemini_topdown/gemini_3_pro_preview_topdown_high.json",
        help="Path to Gemini topdown results (defines training set sample_ids)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to same directory as bagel_file)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(args.bagel_file)

    # Load Gemini topdown to get training sample_ids
    with open(args.gemini_file, "r") as f:
        gemini_data = json.load(f)
    train_sample_ids = {r["sample_id"] for r in gemini_data["results"]}
    print(f"Training sample_ids from Gemini topdown: {len(train_sample_ids)}")

    # Load BAGEL results
    with open(args.bagel_file, "r") as f:
        bagel_data = json.load(f)
    all_results = bagel_data["results"]
    print(f"Total BAGEL results: {len(all_results)}")

    # Split
    train_results = [r for r in all_results if r["sample_id"] in train_sample_ids]
    test_results = [r for r in all_results if r["sample_id"] not in train_sample_ids]

    # Compute accuracy for each split
    def compute_accuracy(results, label):
        if not results:
            print(f"{label}: 0 samples, no accuracy")
            return
        correct = sum(1 for r in results if r.get("accuracy", 0))
        total = len(results)
        print(f"{label}: {correct}/{total} = {correct / total * 100:.2f}% accuracy")

    compute_accuracy(all_results, "All  ")
    compute_accuracy(train_results, "Train")
    compute_accuracy(test_results, "Test ")

    # Check for training ids not found in BAGEL
    bagel_ids = {r["sample_id"] for r in all_results}
    missing = train_sample_ids - bagel_ids
    if missing:
        print(f"Warning: {len(missing)} training sample_ids not found in BAGEL results")

    # Save train split
    train_output = {
        "config": {**bagel_data["config"], "split": "train", "num_samples": len(train_results)},
        "results": train_results,
    }
    train_path = os.path.join(output_dir, "inference_results_bagel_train.json")
    with open(train_path, "w") as f:
        json.dump(train_output, f, indent=2)
    print(f"Saved train split to: {train_path}")

    # Save test split
    test_output = {
        "config": {**bagel_data["config"], "split": "test", "num_samples": len(test_results)},
        "results": test_results,
    }
    test_path = os.path.join(output_dir, "inference_results_bagel_test.json")
    with open(test_path, "w") as f:
        json.dump(test_output, f, indent=2)
    print(f"Saved test split to:  {test_path}")


if __name__ == "__main__":
    main()
