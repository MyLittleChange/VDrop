#!/usr/bin/env python3
"""
Collect all images referenced in a JSONL training file and create a tar.gz archive.

Usage:
    python scripts/collect_training_images.py \
        --jsonl_path /path/to/text_only_thinking.jsonl \
        --image_root_dir /path/to/scratch/spatial_collab_dataset/scenes \
        --output_tar /path/to/training_images.tar.gz
"""

import argparse
import json
import os
import tarfile
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Collect training images into a tar.gz archive"
    )
    parser.add_argument(
        "--jsonl_path",
        default="/path/to/scratch/infinigen/training_data_mix_all_balance/no_thinking/no_thinking.jsonl",
        help="Path to JSONL training data file",
    )
    parser.add_argument(
        "--image_root_dir",
        default="/network/scratch",
        help="Root directory where images are stored (data_dir in dataset_info.py)",
    )
    parser.add_argument(
        "--output_tar",
        default="/path/to/scratch/infinigen/training_data_mix_all_balance/no_thinking/sft_images.tar.gz",
        help="Output tar.gz file path",
    )
    args = parser.parse_args()

    # Collect unique image paths from JSONL
    print(f"Reading JSONL: {args.jsonl_path}")
    rel_paths = set()
    with open(args.jsonl_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            for img in item.get('image', []):
                rel_paths.add(img)

    print(f"Found {len(rel_paths)} unique images")

    # Verify all images exist
    missing = []
    for rel_path in sorted(rel_paths):
        full_path = os.path.join(args.image_root_dir, rel_path)
        if not os.path.exists(full_path):
            missing.append(full_path)

    if missing:
        print(f"WARNING: {len(missing)} images not found:")
        for p in missing[:10]:
            print(f"  {p}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    # Create tar.gz archive preserving relative directory structure
    os.makedirs(os.path.dirname(args.output_tar), exist_ok=True)
    print(f"Creating archive: {args.output_tar}")
    with tarfile.open(args.output_tar, 'w:gz') as tar:
        for rel_path in tqdm(sorted(rel_paths), desc="Archiving"):
            full_path = os.path.join(args.image_root_dir, rel_path)
            if os.path.exists(full_path):
                tar.add(full_path, arcname=rel_path)

    # Print summary
    archive_size = os.path.getsize(args.output_tar)
    size_mb = archive_size / (1024 * 1024)
    print(f"\nDone!")
    print(f"Archive: {args.output_tar}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Images: {len(rel_paths) - len(missing)} archived, {len(missing)} missing")
    print(f"\nTo extract: tar -xzf {args.output_tar} -C <target_dir>")
    print(f"Then set data_dir in dataset_info.py to <target_dir>")


if __name__ == "__main__":
    main()
