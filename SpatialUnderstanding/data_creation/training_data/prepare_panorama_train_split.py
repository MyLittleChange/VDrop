#!/usr/bin/env python3
"""
Prepare panorama training split by collecting samples from V_Final_2000 files
whose scene_id does NOT appear in the test set (approved_mcqs files).

Outputs:
  - panorama_train_samples.json: all non-test samples (for rendering & training)
  - scene_to_sample_id.json: mapping from scene_id to the first sample_id
    (used after rendering to locate each scene's panorama file)

Post-rendering filtering (--filter_rendered):
  After rendering, filter to only samples with successfully rendered panoramas.
  Saves filtered versions as *_filtered.json alongside originals.

Usage:
    # Initial split
    python SpatialUnderstanding/data_creation/training_data/prepare_panorama_train_split.py

    # Filter after rendering
    python SpatialUnderstanding/data_creation/training_data/prepare_panorama_train_split.py \
        --filter_rendered \
        --panorama_dir /path/to/scratch/VisualCoT/infinigen/rendered_panorama_train
"""

import argparse
import json
import os
from collections import OrderedDict


TEST_FILES = [
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_dataset_map_questions_normalized.json"
]

TRAIN_SOURCE_FILES = [
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/anchor_dataset_V_Final_2000.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/counting_dataset_V_Final_2000.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/relative_dataset_V_Final_2000.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/spatial_dataset_V_Final_2000_normalized.json",
]


def filter_rendered(args):
    """Filter panorama_train_samples.json to only keep samples with rendered panoramas."""
    train_path = os.path.join(args.output_dir, "panorama_train_samples.json")
    mapping_path = os.path.join(args.output_dir, "scene_to_sample_id.json")

    print(f"Loading train samples from {train_path}")
    with open(train_path, "r") as f:
        samples = json.load(f)
    print(f"  {len(samples)} samples")

    print(f"Loading scene mapping from {mapping_path}")
    with open(mapping_path, "r") as f:
        scene_to_sample = json.load(f)
    print(f"  {len(scene_to_sample)} scenes")

    # Check which scenes have rendered panoramas
    rendered_scenes = set()
    missing_scenes = []
    for scene_id, first_sample_id in scene_to_sample.items():
        pano_path = os.path.join(
            args.panorama_dir, scene_id,
            f"panorama_blender_limits_{first_sample_id}.png"
        )
        if os.path.exists(pano_path):
            rendered_scenes.add(scene_id)
        else:
            missing_scenes.append(scene_id)

    print(f"\nRendered: {len(rendered_scenes)} / {len(scene_to_sample)} scenes")
    print(f"Missing:  {len(missing_scenes)} scenes")

    # Filter samples
    filtered_samples = [s for s in samples if s["scene_id"] in rendered_scenes]
    filtered_mapping = {k: v for k, v in scene_to_sample.items() if k in rendered_scenes}

    print(f"\nSamples: {len(samples)} → {len(filtered_samples)} (removed {len(samples) - len(filtered_samples)})")
    print(f"Scenes:  {len(scene_to_sample)} → {len(filtered_mapping)}")

    # Save filtered versions
    filtered_train_path = os.path.join(args.output_dir, "panorama_train_samples_filtered.json")
    with open(filtered_train_path, "w") as f:
        json.dump(filtered_samples, f, indent=2)
    print(f"\nFiltered samples saved to: {filtered_train_path}")

    filtered_mapping_path = os.path.join(args.output_dir, "scene_to_sample_id_filtered.json")
    with open(filtered_mapping_path, "w") as f:
        json.dump(filtered_mapping, f, indent=2)
    print(f"Filtered mapping saved to: {filtered_mapping_path}")

    if missing_scenes:
        print(f"\nFirst 10 missing scenes: {missing_scenes[:10]}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare panorama training split (exclude test scenes)"
    )
    parser.add_argument(
        "--filter_rendered", action="store_true",
        help="Filter existing panorama_train_samples.json to only keep samples with rendered panoramas",
    )
    parser.add_argument(
        "--panorama_dir",
        default="/path/to/scratch/VisualCoT/infinigen/rendered_panorama_train",
        help="Directory containing rendered panoramas (for --filter_rendered)",
    )
    parser.add_argument(
        "--test_files", nargs="+", default=TEST_FILES,
        help="Paths to the test set JSON files (approved_mcqs)",
    )
    parser.add_argument(
        "--train_source_files", nargs="+", default=TRAIN_SOURCE_FILES,
        help="Paths to the V_Final_2000 JSON files",
    )
    parser.add_argument(
        "--output_dir",
        default="/path/to/scratch/VisualCoT/training_data",
        help="Output directory",
    )
    args = parser.parse_args()

    if args.filter_rendered:
        filter_rendered(args)
        return

    # Step 1: Collect all test scene_ids
    test_scene_ids = set()
    test_sample_count = 0
    for filepath in args.test_files:
        print(f"Loading test file: {filepath}")
        with open(filepath, "r") as f:
            data = json.load(f)
        for s in data:
            test_scene_ids.add(s["scene_id"])
        test_sample_count += len(data)
        print(f"  {len(data)} samples")

    print(f"\nTest set: {test_sample_count} samples, {len(test_scene_ids)} unique scenes")

    # Step 2: Collect train samples (scene_id NOT in test)
    train_samples = []
    scene_to_first_sample = OrderedDict()  # scene_id → first sample_id seen

    for filepath in args.train_source_files:
        print(f"\nLoading train source: {filepath}")
        with open(filepath, "r") as f:
            data = json.load(f)

        kept = 0
        excluded = 0
        for s in data:
            if s["scene_id"] in test_scene_ids:
                excluded += 1
                continue
            train_samples.append(s)
            kept += 1
            if s["scene_id"] not in scene_to_first_sample:
                scene_to_first_sample[s["scene_id"]] = s["sample_id"]

        print(f"  {len(data)} total, {kept} kept, {excluded} excluded (test scenes)")

    unique_scenes = set(s["scene_id"] for s in train_samples)
    print(f"\nTrain set: {len(train_samples)} samples, {len(unique_scenes)} unique scenes")
    print(f"Test scenes excluded: {len(test_scene_ids)}")

    # Step 3: Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "panorama_train_samples.json")
    with open(train_path, "w") as f:
        json.dump(train_samples, f, indent=2)
    print(f"\nTrain samples saved to: {train_path}")

    # Scene-to-sample mapping (for locating rendered panoramas after rendering)
    # After rendering, panorama path will be:
    #   <render_output_dir>/<scene_id>/panorama_blender_limits_<first_sample_id>.png
    mapping_path = os.path.join(args.output_dir, "scene_to_sample_id.json")
    with open(mapping_path, "w") as f:
        json.dump(scene_to_first_sample, f, indent=2)
    print(f"Scene-to-sample mapping saved to: {mapping_path}")
    print(f"  ({len(scene_to_first_sample)} unique scenes to render)")


if __name__ == "__main__":
    main()
