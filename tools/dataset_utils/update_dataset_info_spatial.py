#!/usr/bin/env python3
"""
Update LLaMA-Factory dataset_info.json with the spatial reasoning dataset.
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Update LLaMA-Factory dataset_info.json'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='SpatialReasoning_VisualCoT',
        help='Name of the dataset'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='/path/to/scratch/VisualCoT/training_data/spatial_reasoning_qwen_sft.json',
        help='Path to the dataset JSON file'
    )
    parser.add_argument(
        '--dataset_info_path',
        type=str,
        default='/path/to/home/LLaMA-Factory/data/dataset_info.json',
        help='Path to LLaMA-Factory dataset_info.json'
    )
    args = parser.parse_args()

    # Load existing dataset info
    dataset_info_path = Path(args.dataset_info_path)
    if dataset_info_path.exists():
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    # Add/update the spatial reasoning dataset
    dataset_info[args.dataset_name] = {
        "file_name": args.dataset_path,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "images": "image"
        }
    }

    # Write back
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=4)

    print(f"Updated {dataset_info_path}")
    print(f"Added dataset: {args.dataset_name}")
    print(f"Dataset path: {args.dataset_path}")


if __name__ == '__main__':
    main()
