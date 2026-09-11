#!/usr/bin/env python3
"""
Prepare panorama rendering split for V4 training scenes.

Collects unique scenes from all 5 V4 normalized datasets, excludes any that
appear in the approved test sets, and writes:
  - panorama_v4_scenes.json      (one entry per unique scene, for rendering)
  - scene_to_sample_id_v4.json   (scene_id → first sample_id, for path resolution)

Post-rendering filtering (--filter_rendered):
  After rendering, filter to only scenes with successfully rendered panoramas.

Usage:
    # Initial split
    python SpatialUnderstanding/data_creation/training_data/prepare_panorama_v4_split.py

    # Filter after rendering
    python SpatialUnderstanding/data_creation/training_data/prepare_panorama_v4_split.py \
        --filter_rendered \
        --panorama_dir /path/to/scratch/infinigen/rendered_panorama_v4
"""

import argparse
import json
import os
from collections import OrderedDict


V4_TRAIN_FILES = [
    "/path/to/scratch/infinigen/dataset_counting_questions_filtered_V4_normalized.json",
    "/path/to/scratch/infinigen/dataset_anchor_questions_filtered_V4_normalized.json",
    "/path/to/scratch/infinigen/dataset_spatial_questions_filtered_V4_normalized.json",
    "/path/to/scratch/infinigen/dataset_relative_distance_questions_filtered_V4_normalized.json",
    "/path/to/scratch/infinigen/dataset_perspective_taking_questions_filtered_V4_normalized.json",
]

TEST_FILES = [
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_dataset_map_questions_normalized.json",
]

OUTPUT_DIR = "/path/to/scratch/infinigen"
# Base path for V4 scenes (blend files at <BLEND_BASE>/<room_part>/<scene_id>/fine/scene.blend)
V4_BLEND_BASE = "/path/to/scratch/infinigen/outputs_rendered"


def filter_rendered(args):
    """Filter panorama_v4_scenes.json to only keep scenes with rendered panoramas."""
    scenes_path = os.path.join(args.output_dir, "panorama_v4_scenes.json")
    mapping_path = os.path.join(args.output_dir, "scene_to_sample_id_v4.json")

    print(f"Loading scenes from {scenes_path}")
    with open(scenes_path, "r") as f:
        samples = json.load(f)
    print(f"  {len(samples)} unique scenes")

    print(f"Loading mapping from {mapping_path}")
    with open(mapping_path, "r") as f:
        scene_to_sample = json.load(f)

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

    filtered_samples = [s for s in samples if s["scene_id"] in rendered_scenes]
    filtered_mapping = {k: v for k, v in scene_to_sample.items() if k in rendered_scenes}

    filtered_scenes_path = os.path.join(args.output_dir, "panorama_v4_scenes_filtered.json")
    with open(filtered_scenes_path, "w") as f:
        json.dump(filtered_samples, f, indent=2)
    print(f"\nFiltered scenes saved to: {filtered_scenes_path}")

    filtered_mapping_path = os.path.join(args.output_dir, "scene_to_sample_id_v4_filtered.json")
    with open(filtered_mapping_path, "w") as f:
        json.dump(filtered_mapping, f, indent=2)
    print(f"Filtered mapping saved to: {filtered_mapping_path}")

    if missing_scenes:
        print(f"\nFirst 10 missing scenes: {missing_scenes[:10]}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare panorama split for V4 training scenes"
    )
    parser.add_argument(
        "--filter_rendered", action="store_true",
        help="Filter existing panorama_v4_scenes.json to only scenes with rendered panoramas",
    )
    parser.add_argument(
        "--panorama_dir",
        default="/path/to/scratch/infinigen/rendered_panorama_v4",
        help="Directory containing rendered panoramas (for --filter_rendered)",
    )
    parser.add_argument(
        "--v4_train_files", nargs="+", default=V4_TRAIN_FILES,
        help="Paths to V4 normalized training JSONs",
    )
    parser.add_argument(
        "--test_files", nargs="+", default=TEST_FILES,
        help="Paths to approved normalized test JSONs",
    )
    parser.add_argument(
        "--output_dir", default=OUTPUT_DIR,
        help="Output directory for scene list and mapping",
    )
    args = parser.parse_args()

    if args.filter_rendered:
        filter_rendered(args)
        return

    # Collect test scene IDs to exclude
    test_scene_ids = set()
    for filepath in args.test_files:
        if not os.path.exists(filepath):
            print(f"[SKIP] Test file not found: {filepath}")
            continue
        with open(filepath, "r") as f:
            data = json.load(f)
        for s in data:
            test_scene_ids.add(s["scene_id"])
    print(f"Test scenes to exclude: {len(test_scene_ids)}")

    # Collect unique V4 scenes (one sample per scene for rendering)
    scene_to_first_sample = OrderedDict()  # scene_id → first sample seen
    all_unique_samples = []  # one sample per unique scene
    total_samples = 0
    total_excluded = 0

    for filepath in args.v4_train_files:
        if not os.path.exists(filepath):
            print(f"[SKIP] V4 file not found: {filepath}")
            continue
        with open(filepath, "r") as f:
            data = json.load(f)
        total_samples += len(data)
        excluded = 0
        for s in data:
            scene_id = s["scene_id"]
            if scene_id in test_scene_ids:
                excluded += 1
                total_excluded += 1
                continue
            if scene_id not in scene_to_first_sample:
                scene_to_first_sample[scene_id] = s["sample_id"]
                # Add blend_base so render script can find fine/scene.blend
                s["blend_base"] = V4_BLEND_BASE
                s["blend_subdir"] = "fine"
                all_unique_samples.append(s)
        print(f"{os.path.basename(filepath)}: {len(data)} samples, {excluded} excluded (test), {len(data)-excluded} kept")

    print(f"\nTotal samples across all V4 files: {total_samples}")
    print(f"Excluded (test scenes): {total_excluded}")
    print(f"Unique scenes for rendering: {len(all_unique_samples)}")

    os.makedirs(args.output_dir, exist_ok=True)

    scenes_path = os.path.join(args.output_dir, "panorama_v4_scenes.json")
    with open(scenes_path, "w") as f:
        json.dump(all_unique_samples, f, indent=2)
    print(f"\nScene list saved to: {scenes_path}")
    print(f"  ({len(all_unique_samples)} unique scenes to render)")

    mapping_path = os.path.join(args.output_dir, "scene_to_sample_id_v4.json")
    with open(mapping_path, "w") as f:
        json.dump(dict(scene_to_first_sample), f, indent=2)
    print(f"Scene-to-sample mapping saved to: {mapping_path}")


if __name__ == "__main__":
    main()
