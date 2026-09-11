#!/usr/bin/env python3
"""
Normalize V4 filtered QA datasets to always reference "second image" perspective,
and add topdown_path/panorama_path fields (set to None, filled after rendering).

Processes all 5 V4 filtered datasets:
  - dataset_counting_questions_filtered_V4.json
  - dataset_anchor_questions_filtered_V4.json
  - dataset_spatial_questions_filtered_V4.json
  - dataset_relative_distance_questions_filtered_V4.json
  - dataset_perspective_taking_questions_filtered_V4.json

Usage:
    python tools/dataset_utils/normalize_v4_datasets.py
    python tools/dataset_utils/normalize_v4_datasets.py --input_dir /path/to/dir --output_dir /path/to/out
"""

import argparse
import json
import os


V4_DATASETS = [
    "dataset_counting_questions_filtered_V4.json",
    "dataset_anchor_questions_filtered_V4.json",
    "dataset_spatial_questions_filtered_V4.json",
    "dataset_relative_distance_questions_filtered_V4.json",
    "dataset_perspective_taking_questions_filtered_V4.json",
]

DEFAULT_DIR = "/path/to/scratch/infinigen"


def remap_path(path: str) -> str:
    """Remap paths from home directories to network scratch."""
    if path is None:
        return None
    path = path.replace(
        "/path/to/scratch",
        "/path/to/scratch",
    )
    path = path.replace(
        "/path/to/scratch",
        "/path/to/scratch",
    )
    return path


def normalize_dataset(data: list) -> tuple[list, int, int]:
    """
    Normalize samples: remap paths and add panorama/topdown fields.

    Note: image order is NOT swapped — user_1_image_local_path is always camera_0,
    which is what question_both_views refers to as "first image".

    Returns (normalized_data, 0, len(data)) for API compatibility.
    """
    for sample in data:
        # Remap paths (already on scratch but apply for safety)
        sample["user_1_image_local_path"] = remap_path(sample.get("user_1_image_local_path"))
        sample["user_2_image_local_path"] = remap_path(sample.get("user_2_image_local_path"))

        # Add panorama/topdown fields (filled after rendering)
        if "panorama_path" not in sample:
            sample["panorama_path"] = None
        if "topdown_path" not in sample:
            sample["topdown_path"] = None

    return data, 0, len(data)


def main():
    parser = argparse.ArgumentParser(description="Normalize V4 filtered QA datasets")
    parser.add_argument(
        "--input_dir",
        default=DEFAULT_DIR,
        help="Directory containing dataset_*_filtered_V4.json files",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_DIR,
        help="Output directory for normalized files (default: same as input)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_in = 0
    total_out = 0

    for filename in V4_DATASETS:
        input_path = os.path.join(args.input_dir, filename)
        output_filename = filename.replace("_V4.json", "_V4_normalized.json")
        output_path = os.path.join(args.output_dir, output_filename)

        if not os.path.exists(input_path):
            print(f"[SKIP] Not found: {input_path}")
            continue

        with open(input_path, "r") as f:
            data = json.load(f)

        print(f"\n{filename}: {len(data)} samples")
        data, swapped, already = normalize_dataset(data)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  Swapped: {swapped}, already second: {already}")
        print(f"  Saved to: {output_path}")
        total_in += len(data)
        total_out += len(data)

    print(f"\nDone. Total: {total_in} samples normalized.")


if __name__ == "__main__":
    main()
