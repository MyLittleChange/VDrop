#!/usr/bin/env python3
"""
Prepare panorama rendering split for V5 training scenes (spatial/).

Key difference from V4: blend files vary per scene — some have fine/scene.blend,
others only have coarse/scene.blend. For each scene we check fine first, then
fall back to coarse.

Collects unique scenes from all 5 V5 normalized datasets, excludes any that
appear in the approved test sets, and writes:
  - panorama_v5_scenes.json      (one entry per unique scene, for rendering)
  - scene_to_sample_id_v5.json   (scene_id → first sample_id, for path resolution)

Post-rendering filtering (--filter_rendered):
  After rendering, filter to only scenes with successfully rendered panoramas.

Usage:
    # Initial split
    python SpatialUnderstanding/data_creation/training_data/prepare_panorama_v5_split.py

    # Filter after rendering
    python SpatialUnderstanding/data_creation/training_data/prepare_panorama_v5_split.py \
        --filter_rendered \
        --panorama_dir /path/to/scratch/infinigen/rendered_panorama_v5
"""

import argparse
import json
import os
from collections import OrderedDict


V5_TRAIN_FILES = [
    "/path/to/scratch/infinigen/spatial/dataset_counting_questions_filtered_V5_normalized.json",
    "/path/to/scratch/infinigen/spatial/dataset_anchor_questions_filtered_V5_normalized.json",
    "/path/to/scratch/infinigen/spatial/dataset_spatial_questions_filtered_V5_normalized.json",
    "/path/to/scratch/infinigen/spatial/dataset_relative_distance_questions_filtered_V5_normalized.json",
    "/path/to/scratch/infinigen/spatial/dataset_perspective_taking_questions_filtered_V5_normalized.json",
]

TEST_FILES = [
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_dataset_map_questions_normalized.json",
]

OUTPUT_DIR = "/path/to/scratch/infinigen"
# Base path for V5 scenes (blend files at spatial/<room_part>/<scene_id>/{fine,coarse}/scene.blend)
V5_BLEND_BASE = "/path/to/scratch/infinigen/spatial"


def resolve_blend_subdir(blend_base: str, room_part: str, scene_id: str) -> str:
    """Check fine/scene.blend first, fall back to coarse/scene.blend."""
    for subdir in ("fine", "coarse"):
        blend_path = os.path.join(blend_base, room_part, scene_id, subdir, "scene.blend")
        if os.path.exists(blend_path):
            return subdir
    return "coarse"  # default fallback


def extract_room_part(sample: dict) -> str:
    """Extract room_part from sample's image path (e.g., LivingRoom_v201_Part4).

    The path looks like: .../spatial/LivingRoom_v201_Part4/<scene_hash>/frames/...
    """
    # Try to get room_part from the sample directly
    if "room_part" in sample and sample["room_part"]:
        return sample["room_part"]

    # Extract from image path
    img_path = sample.get("user_1_image_local_path") or sample.get("user_2_image_local_path", "")
    parts = img_path.split("/")
    # Find "spatial" in path, room_part is the next component
    for i, part in enumerate(parts):
        if part == "spatial" and i + 1 < len(parts):
            return parts[i + 1]
    return ""


def filter_rendered(args):
    """Filter panorama_v5_scenes.json to only keep scenes with rendered panoramas."""
    scenes_path = os.path.join(args.output_dir, "panorama_v5_scenes.json")
    mapping_path = os.path.join(args.output_dir, "scene_to_sample_id_v5.json")

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

    filtered_scenes_path = os.path.join(args.output_dir, "panorama_v5_scenes_filtered.json")
    with open(filtered_scenes_path, "w") as f:
        json.dump(filtered_samples, f, indent=2)
    print(f"\nFiltered scenes saved to: {filtered_scenes_path}")

    filtered_mapping_path = os.path.join(args.output_dir, "scene_to_sample_id_v5_filtered.json")
    with open(filtered_mapping_path, "w") as f:
        json.dump(filtered_mapping, f, indent=2)
    print(f"Filtered mapping saved to: {filtered_mapping_path}")

    if missing_scenes:
        print(f"\nFirst 10 missing scenes: {missing_scenes[:10]}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare panorama split for V5 training scenes"
    )
    parser.add_argument(
        "--filter_rendered", action="store_true",
        help="Filter existing panorama_v5_scenes.json to only scenes with rendered panoramas",
    )
    parser.add_argument(
        "--panorama_dir",
        default="/path/to/scratch/infinigen/rendered_panorama_v5",
        help="Directory containing rendered panoramas (for --filter_rendered)",
    )
    parser.add_argument(
        "--v5_train_files", nargs="+", default=V5_TRAIN_FILES,
        help="Paths to V5 normalized training JSONs",
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

    # Collect unique V5 scenes (one sample per scene for rendering)
    scene_to_first_sample = OrderedDict()  # scene_id → first sample seen
    all_unique_samples = []  # one sample per unique scene
    total_samples = 0
    total_excluded = 0
    blend_subdir_counts = {"fine": 0, "coarse": 0}

    for filepath in args.v5_train_files:
        if not os.path.exists(filepath):
            print(f"[SKIP] V5 file not found: {filepath}")
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

                room_part = extract_room_part(s)
                blend_subdir = resolve_blend_subdir(V5_BLEND_BASE, room_part, scene_id)
                blend_subdir_counts[blend_subdir] += 1

                s["blend_base"] = V5_BLEND_BASE
                s["blend_subdir"] = blend_subdir
                s["room_part"] = room_part
                all_unique_samples.append(s)
        print(f"{os.path.basename(filepath)}: {len(data)} samples, {excluded} excluded (test), {len(data)-excluded} kept")

    print(f"\nTotal samples across all V5 files: {total_samples}")
    print(f"Excluded (test scenes): {total_excluded}")
    print(f"Unique scenes for rendering: {len(all_unique_samples)}")
    print(f"Blend subdirs: fine={blend_subdir_counts['fine']}, coarse={blend_subdir_counts['coarse']}")

    os.makedirs(args.output_dir, exist_ok=True)

    scenes_path = os.path.join(args.output_dir, "panorama_v5_scenes.json")
    with open(scenes_path, "w") as f:
        json.dump(all_unique_samples, f, indent=2)
    print(f"\nScene list saved to: {scenes_path}")
    print(f"  ({len(all_unique_samples)} unique scenes to render)")

    mapping_path = os.path.join(args.output_dir, "scene_to_sample_id_v5.json")
    with open(mapping_path, "w") as f:
        json.dump(dict(scene_to_first_sample), f, indent=2)
    print(f"Scene-to-sample mapping saved to: {mapping_path}")


if __name__ == "__main__":
    main()
