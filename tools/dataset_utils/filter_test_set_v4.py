#!/usr/bin/env python3
"""
Filter approved normalized test sets to remove any scenes that appear in V4 training data.

Currently there is 0 overlap between V4 (outputs_rendered scenes) and the approved
test sets (ankur's scenes). This script confirms that and writes filtered copies.

Usage:
    python tools/dataset_utils/filter_test_set_v4.py
"""

import argparse
import json
import os


V4_TRAIN_FILES = [
    "/path/to/scratch/infinigen/dataset_counting_questions_filtered_V4.json",
    "/path/to/scratch/infinigen/dataset_anchor_questions_filtered_V4.json",
    "/path/to/scratch/infinigen/dataset_spatial_questions_filtered_V4.json",
    "/path/to/scratch/infinigen/dataset_relative_distance_questions_filtered_V4.json",
    "/path/to/scratch/infinigen/dataset_perspective_taking_questions_filtered_V4.json",
]

TEST_FILES = [
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_dataset_map_questions_normalized.json",
]


def main():
    parser = argparse.ArgumentParser(
        description="Filter approved test sets to remove V4 training scenes"
    )
    parser.add_argument(
        "--v4_train_files", nargs="+", default=V4_TRAIN_FILES,
        help="Paths to V4 filtered training JSONs",
    )
    parser.add_argument(
        "--test_files", nargs="+", default=TEST_FILES,
        help="Paths to approved normalized test JSONs",
    )
    parser.add_argument(
        "--output_suffix", default="_filtered_v4",
        help="Suffix to add before .json in output filenames",
    )
    args = parser.parse_args()

    # Collect all V4 training scene IDs
    v4_scenes = set()
    for filepath in args.v4_train_files:
        if not os.path.exists(filepath):
            print(f"[SKIP] V4 file not found: {filepath}")
            continue
        with open(filepath, "r") as f:
            data = json.load(f)
        scenes = {s["scene_id"] for s in data}
        v4_scenes |= scenes
        print(f"V4 train: {os.path.basename(filepath)} → {len(data)} samples, {len(scenes)} scenes")

    print(f"\nTotal V4 training scenes: {len(v4_scenes)}")

    # Filter each test file
    print()
    total_overlap = 0
    for filepath in args.test_files:
        if not os.path.exists(filepath):
            print(f"[SKIP] Test file not found: {filepath}")
            continue

        with open(filepath, "r") as f:
            data = json.load(f)

        test_scenes = {s["scene_id"] for s in data}
        overlap = test_scenes & v4_scenes
        total_overlap += len(overlap)

        filtered = [s for s in data if s["scene_id"] not in v4_scenes]
        removed = len(data) - len(filtered)

        # Write output
        base, ext = os.path.splitext(filepath)
        output_path = base + args.output_suffix + ext
        with open(output_path, "w") as f:
            json.dump(filtered, f, indent=2)

        print(f"{os.path.basename(filepath)}: {len(data)} → {len(filtered)} samples (removed {removed}, overlap scenes: {len(overlap)})")
        print(f"  Saved to: {output_path}")

    print(f"\nTotal overlap scenes found across all test files: {total_overlap}")
    if total_overlap == 0:
        print("✓ No overlap — test sets are clean.")


if __name__ == "__main__":
    main()
