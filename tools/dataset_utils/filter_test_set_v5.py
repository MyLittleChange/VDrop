#!/usr/bin/env python3
"""
Filter approved normalized test sets to remove any scenes that appear in V5 training data.

V5 scenes come from /path/to/scratch/infinigen/spatial/.
Expected 0 overlap with the approved test sets (ankur's scenes), but this script
confirms that and writes filtered copies.

Usage:
    python tools/dataset_utils/filter_test_set_v5.py
"""

import argparse
import json
import os


V5_TRAIN_FILES = [
    "/path/to/scratch/infinigen/spatial/dataset_counting_questions_filtered_V5.json",
    "/path/to/scratch/infinigen/spatial/dataset_anchor_questions_filtered_V5.json",
    "/path/to/scratch/infinigen/spatial/dataset_spatial_questions_filtered_V5.json",
    "/path/to/scratch/infinigen/spatial/dataset_relative_distance_questions_filtered_V5.json",
    "/path/to/scratch/infinigen/spatial/dataset_perspective_taking_questions_filtered_V5.json",
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
        description="Filter approved test sets to remove V5 training scenes"
    )
    parser.add_argument(
        "--v5_train_files", nargs="+", default=V5_TRAIN_FILES,
        help="Paths to V5 filtered training JSONs",
    )
    parser.add_argument(
        "--test_files", nargs="+", default=TEST_FILES,
        help="Paths to approved normalized test JSONs",
    )
    parser.add_argument(
        "--output_suffix", default="_filtered_v5",
        help="Suffix to add before .json in output filenames",
    )
    args = parser.parse_args()

    # Collect all V5 training scene IDs
    v5_scenes = set()
    for filepath in args.v5_train_files:
        if not os.path.exists(filepath):
            print(f"[SKIP] V5 file not found: {filepath}")
            continue
        with open(filepath, "r") as f:
            data = json.load(f)
        scenes = {s["scene_id"] for s in data}
        v5_scenes |= scenes
        print(f"V5 train: {os.path.basename(filepath)} → {len(data)} samples, {len(scenes)} scenes")

    print(f"\nTotal V5 training scenes: {len(v5_scenes)}")

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
        overlap = test_scenes & v5_scenes
        total_overlap += len(overlap)

        filtered = [s for s in data if s["scene_id"] not in v5_scenes]
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
