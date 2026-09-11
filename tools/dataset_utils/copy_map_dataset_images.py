#!/usr/bin/env python3
"""
Copy input images (user_1, user_2) and map_questions3 map images
for samples in map_filtered_final_correctness_list.json to a local working directory.

Usage:
    python scripts/copy_map_dataset_images.py
    python scripts/copy_map_dataset_images.py --dry_run   # preview without copying
"""

import argparse
import json
import os
import shutil
from pathlib import Path

JSON_FILE = "/path/to/scratch/spatial_collab_dataset/map_filtered_final_correctness_list.json"
V00_BASE = Path("/path/to/scratch/infinigen/infinigen_debang/infinigen/v00_filtered")
OUTPUT_DIR = Path("/path/to/ThinkMorph-BAGEL-release/map_dataset_images")


def get_version_folder(sample: dict) -> str | None:
    """Extract version folder (e.g. 'v7_dataset_filtered') from image path."""
    img_path = sample.get("user_1_image_local_path", "")
    for part in img_path.split("/"):
        if part.startswith("v") and "dataset_filtered" in part:
            return part
    return None


def get_map_questions3_dir(sample: dict, version_folder: str) -> Path:
    """Build the map_questions3 directory path for a sample."""
    return V00_BASE / version_folder / sample["room_part"] / sample["scene_id"] / "map_questions3"


def get_map_filename(sample: dict) -> str:
    """Extract the map image filename from map_image_path."""
    return os.path.basename(sample["map_image_path"])


def copy_file(src: str | Path, dst: str | Path, dry_run: bool = False) -> bool:
    """Copy a file, creating parent dirs as needed. Returns True if successful."""
    src, dst = Path(src), Path(dst)
    if not src.exists():
        return False
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True


def main():
    parser = argparse.ArgumentParser(description="Copy map dataset images to working dir")
    parser.add_argument("--json_file", type=str, default=JSON_FILE)
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--dry_run", action="store_true", help="Preview without copying")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    with open(args.json_file) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} samples from {args.json_file}")

    stats = {"total": len(dataset), "processed": 0, "skipped_no_version": 0,
             "skipped_no_mq3": 0, "input_copied": 0, "map_copied": 0,
             "input_missing": 0, "map_missing": 0}

    for sample in dataset:
        sample_id = sample["sample_id"]
        scene_id = sample["scene_id"]
        version_folder = get_version_folder(sample)

        if version_folder is None:
            stats["skipped_no_version"] += 1
            continue

        mq3_dir = get_map_questions3_dir(sample, version_folder)
        if not mq3_dir.is_dir():
            stats["skipped_no_mq3"] += 1
            continue

        stats["processed"] += 1

        # Output structure: output_dir/<scene_id>/<sample_id>/
        sample_out = output_dir / scene_id / sample_id

        # --- Copy input images (user_1 and user_2) ---
        for key in ("user_1_image_local_path", "user_2_image_local_path"):
            src = sample.get(key, "")
            if not src:
                continue
            filename = os.path.basename(src)
            tag = "user1" if "user_1" in key else "user2"
            dst = sample_out / f"{tag}_{filename}"
            if copy_file(src, dst, dry_run=args.dry_run):
                stats["input_copied"] += 1
            else:
                stats["input_missing"] += 1
                print(f"  WARNING: input image missing: {src}")

        # --- Copy map image from map_questions3 ---
        map_filename = get_map_filename(sample)
        map_src = mq3_dir / map_filename
        map_dst = sample_out / f"map3_{map_filename}"
        if copy_file(map_src, map_dst, dry_run=args.dry_run):
            stats["map_copied"] += 1
        else:
            stats["map_missing"] += 1
            print(f"  WARNING: map_questions3 image missing: {map_src}")

    # Print summary
    print("\n===== Summary =====")
    print(f"Total samples:          {stats['total']}")
    print(f"Processed (has mq3):    {stats['processed']}")
    print(f"Skipped (no version):   {stats['skipped_no_version']}")
    print(f"Skipped (no mq3 dir):   {stats['skipped_no_mq3']}")
    print(f"Input images copied:    {stats['input_copied']}")
    print(f"Input images missing:   {stats['input_missing']}")
    print(f"Map images copied:      {stats['map_copied']}")
    print(f"Map images missing:     {stats['map_missing']}")
    if args.dry_run:
        print("\n[DRY RUN] No files were actually copied.")
    else:
        print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
