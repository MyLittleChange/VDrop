#!/usr/bin/env python3
"""
Convert parsed QA annotation JSON to training dataset format (parquet files or JSONL).

This script reads parsed QA annotations and converts them to the training format
used by ThinkMorph.

Supports three thinking modes:
  --thinking_mode interleaved_thinking  (default)
    Text reasoning + generated image.
    Images: [image1, image2, center_view]
    Output: [<think>thinking_annotation</think><image_start>, <image_end>]

  --thinking_mode text_only_thinking
    Text reasoning only (no generated reasoning image).
    Images: [image1, image2]
    Output: [<think>thinking_annotation</think>]

  --thinking_mode visual_only_thinking
    Visual thinking only (no text reasoning).
    Images: [image1, image2, center_view]
    Output: [<image_start>, <image_end>]

Image mapping (supports both naming conventions):
- image1: First context image (image1_path or image0_path)
- image2: Second context image (image2_path)
- center_view: Target view image for reasoning (center_view_path or target_path)
"""

import argparse
import json
import os
import io
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import pyarrow as pa
import pyarrow.parquet as pq


# --- System prompts for each thinking mode ---

INTERLEAVED_THINK_SYSTEM_PROMPT = (
    "Let's think step by step to answer the question. "
    "For text-based thinking, enclose the process within <think> </think>. "
    "For visual thinking, enclose the content within <image_start> </image_end>."
)

TEXT_ONLY_THINK_SYSTEM_PROMPT = (
    "Let's think step by step to answer the question. "
    "Enclose your thinking process within <think> </think> tags."
)

VISUAL_ONLY_THINK_SYSTEM_PROMPT = (
    "Think visually to answer the question. "
    "Enclose your visual thinking within <image_start> </image_end>."
)


def image_to_bytes(image_path: str) -> bytes:
    """Load image from path and convert to PNG bytes."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            byte_stream = io.BytesIO()
            img.save(byte_stream, format='PNG')
            return byte_stream.getvalue()
    except Exception as e:
        print(f"[ERROR] Failed to load image {image_path}: {e}")
        return None


def remap_path(path: str) -> str:
    """Remap paths from home directories to network scratch."""
    if path is None:
        return None
    path = path.replace(
        "/path/to/scratch",
        "/path/to/scratch"
    )
    path = path.replace(
        "/path/to/scratch",
        "/path/to/scratch"
    )
    return path


def get_image_paths(result: dict):
    """Resolve image paths, supporting both naming conventions."""
    image1_path = remap_path(result.get('image1_path') or result.get('image0_path'))
    image2_path = remap_path(result.get('image2_path'))
    center_view_path = remap_path(result.get('center_view_path') or result.get('target_path'))
    return image1_path, image2_path, center_view_path


def process_item_interleaved(result: dict):
    """Interleaved thinking: text + image."""
    try:
        sample_id = result.get('sample_id', '')
        question = result.get('question', '')
        thinking_annotation = result.get('thinking_annotation', '')

        if not question or not thinking_annotation:
            print(f"[WARNING] Missing question or thinking for sample {sample_id}")
            return None

        # Get image paths
        image1_path, image2_path, center_view_path = get_image_paths(result)

        if not all([image1_path, image2_path, center_view_path]):
            print(f"[WARNING] Missing image paths for sample {sample_id}")
            return None

        # Load images
        img1_bytes = image_to_bytes(image1_path)
        img2_bytes = image_to_bytes(image2_path)
        center_view_bytes = image_to_bytes(center_view_path)

        if not all([img1_bytes, img2_bytes, center_view_bytes]):
            print(f"[WARNING] Failed to load images for sample {sample_id}")
            return None

        return {
            "image_list": [img1_bytes, img2_bytes, center_view_bytes],
            "instruction_list": [INTERLEAVED_THINK_SYSTEM_PROMPT + "\n" + question],
            "output_text_list": [
                "<think>" + thinking_annotation + "</think><image_start>",
                "<image_end>"
            ],
            "sample_id": sample_id,
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {result.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_item_text_only(result: dict, image_root_dir: str = "/network/scratch"):
    """Text-only thinking: ShareGPT/JSONL format for vlm_sft dataset.

    Returns dict with conversations (ShareGPT) + relative image paths.
    Images are referenced via <image> tokens and loaded by SftJSONLIterableDataset
    which does os.path.join(data_dir, image_path).
    """
    try:
        sample_id = result.get('sample_id', '')
        question = result.get('question', '')
        thinking_annotation = result.get('thinking_annotation', '')

        if not question or not thinking_annotation:
            print(f"[WARNING] Missing question or thinking for sample {sample_id}")
            return None

        # Get image paths
        image1_path, image2_path, _ = get_image_paths(result)

        if not all([image1_path, image2_path]):
            print(f"[WARNING] Missing image paths for sample {sample_id}")
            return None

        # Verify images exist
        for img_path in [image1_path, image2_path]:
            if not os.path.exists(img_path):
                print(f"[WARNING] Image not found: {img_path}")
                return None

        # Convert absolute paths to relative paths w.r.t. image_root_dir
        rel_img1 = os.path.relpath(image1_path, image_root_dir)
        rel_img2 = os.path.relpath(image2_path, image_root_dir)

        # Build ShareGPT format: <image> tokens reference images in order
        user_message = f"{TEXT_ONLY_THINK_SYSTEM_PROMPT}\n\n<image><image>\n{question}"
        assistant_response = f"<think>{thinking_annotation}</think>"

        return {
            "conversations": [
                {"from": "human", "value": user_message},
                {"from": "gpt", "value": assistant_response}
            ],
            "image": [rel_img1, rel_img2],
            "id": sample_id,
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {result.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_item_visual_only(result: dict):
    """Visual-only thinking: reasoning image only (no text reasoning), then answer."""
    try:
        sample_id = result.get('sample_id', '')
        question = result.get('question', '')

        if not question:
            print(f"[WARNING] Missing question for sample {sample_id}")
            return None

        # Get image paths
        image1_path, image2_path, center_view_path = get_image_paths(result)

        if not all([image1_path, image2_path, center_view_path]):
            print(f"[WARNING] Missing image paths for sample {sample_id}")
            return None

        # Load images
        img1_bytes = image_to_bytes(image1_path)
        img2_bytes = image_to_bytes(image2_path)
        center_view_bytes = image_to_bytes(center_view_path)

        if not all([img1_bytes, img2_bytes, center_view_bytes]):
            print(f"[WARNING] Failed to load images for sample {sample_id}")
            return None

        return {
            "image_list": [img1_bytes, img2_bytes, center_view_bytes],
            "instruction_list": [VISUAL_ONLY_THINK_SYSTEM_PROMPT + "\n" + question],
            "output_text_list": [
                "<image_start>",
                "<image_end>"
            ],
            "sample_id": sample_id,
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {result.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


THINKING_MODE_PROCESSORS = {
    "interleaved_thinking": process_item_interleaved,
    "text_only_thinking": process_item_text_only,
    "visual_only_thinking": process_item_visual_only,
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert parsed QA annotations to training dataset format"
    )
    parser.add_argument(
        "--annotation_file",
        nargs="+",
        default=[
            "/path/to/scratch/VisualCoT/annotations/gemini_3_flash_preview_rendered_orbit_orbit_annotations_parsed_qa_train.json"
            # "/path/to/scratch/VisualCoT/annotations/gemini_3_flash_preview_rendered_v3_anchor_annotations_center_view_no_angle_parsed_qa.json",
            # "/path/to/scratch/VisualCoT/annotations/gemini_3_flash_preview_rendered_v4_spatial_annotations_rear_view_parsed_qa.json",
        ],
        help="Path(s) to the parsed QA annotation JSON file(s)"
    )
    parser.add_argument(
        "--output_dir",
        default="/path/to/scratch/VisualCoT/training_data/orbit_orbit_annotations_parsed_qa_train_visual_only_thinking",
        help="Output directory for parquet files or JSONL"
    )
    parser.add_argument(
        "--thinking_mode",
        choices=["interleaved_thinking", "text_only_thinking", "visual_only_thinking"],
        default="visual_only_thinking",
        help=(
            "Thinking mode for training data: "
            "interleaved_thinking = text + image (default), "
            "text_only_thinking = text reasoning only (no reasoning image), "
            "visual_only_thinking = visual reasoning only (no text reasoning)"
        ),
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of parallel workers for processing"
    )
    parser.add_argument(
        "--rows_per_group",
        type=int,
        default=10,
        help="Rows per row group in parquet files"
    )
    parser.add_argument(
        "--groups_per_file",
        type=int,
        default=10,
        help="Row groups per parquet file"
    )
    parser.add_argument(
        "--image_root_dir",
        default="/path/to/scratch/VisualCoT/infinigen/",
        help=(
            "Root directory for images (text_only_thinking mode only). "
            "Image paths in JSONL will be relative to this directory. "
            "Must match data_dir in dataset_info.py."
        ),
    )

    args = parser.parse_args()

    process_fn = THINKING_MODE_PROCESSORS[args.thinking_mode]
    if args.thinking_mode == "text_only_thinking":
        process_fn = partial(process_fn, image_root_dir=args.image_root_dir)
    print(f"Thinking mode: {args.thinking_mode}")

    # Load annotations from all files
    results = []
    for annotation_file in args.annotation_file:
        print(f"Loading annotations from {annotation_file}")
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        file_results = data.get('results', [])
        print(f"  -> {len(file_results)} results")
        results.extend(file_results)
    print(f"Loaded {len(results)} total annotation results")

    # Filter out empty questions (and thinking annotations for non-visual-only modes)
    if args.thinking_mode == "visual_only_thinking":
        results = [r for r in results if r.get('question', '').strip()]
        print(f"After filtering empty questions: {len(results)} samples")
    else:
        results = [r for r in results if r.get('question', '').strip() and r.get('thinking_annotation', '').strip()]
        print(f"After filtering empty questions/thinking: {len(results)} samples")

    # Process items in parallel
    print(f"Processing samples with {args.max_workers} workers...")
    all_data = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        processed = list(tqdm(executor.map(process_fn, results), total=len(results)))
        all_data = [res for res in processed if res is not None]

    random.shuffle(all_data)
    print(f"Successfully processed {len(all_data)} samples")

    if len(all_data) == 0:
        print("No samples to write. Exiting.")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.thinking_mode == "text_only_thinking":
        # Write JSONL for vlm_sft dataset (ShareGPT format)
        jsonl_file = os.path.join(args.output_dir, "text_only_thinking.jsonl")
        print(f"Writing JSONL to {jsonl_file}")
        with open(jsonl_file, 'w') as f:
            for item in all_data:
                f.write(json.dumps(item) + '\n')

        print(f"\nDone! Created {jsonl_file}")
        print(f"Total samples: {len(all_data)}")

        # Print sample for verification
        sample = all_data[0]
        print("\n--- Sample entry ---")
        print(f"ID: {sample['id']}")
        print(f"Images: {sample['image']}")
        print(f"Human: {sample['conversations'][0]['value'][:200]}...")
        print(f"GPT: {sample['conversations'][1]['value'][:200]}...")
    else:
        # Write to parquet files for interleaved_thinking and visual_only_thinking modes
        # Write to parquet files for interleaved_thinking mode
        rows_per_file = args.rows_per_group * args.groups_per_file
        parquet_base_name = "chunk_"
        file_index = 0

        # Remove sample_id before writing (only for training data format)
        for item in all_data:
            if 'sample_id' in item:
                del item['sample_id']

        # Track parquet info for each file
        parquet_info = {}

        print(f"Writing parquet files to {args.output_dir}")
        for i in tqdm(range(0, len(all_data), rows_per_file)):
            file_data = all_data[i:i + rows_per_file]
            parquet_file = os.path.join(args.output_dir, f"{parquet_base_name}{file_index}.parquet")
            file_index += 1

            schema = pa.schema([
                pa.field("image_list", pa.list_(pa.binary())),
                pa.field("instruction_list", pa.list_(pa.string())),
                pa.field("output_text_list", pa.list_(pa.string())),
            ])

            num_row_groups = 0
            with pq.ParquetWriter(parquet_file, schema=schema, version="2.6") as writer:
                for j in range(0, len(file_data), args.rows_per_group):
                    group_data = file_data[j:j + args.rows_per_group]
                    group_table = pa.Table.from_pylist(group_data, schema=schema)
                    writer.write_table(group_table)
                    num_row_groups += 1

            # Record parquet info for this file
            parquet_info[parquet_file] = {
                "num_row_groups": num_row_groups,
                "num_rows": len(file_data),
            }

        # Write parquet_info.json
        parquet_info_path = os.path.join(args.output_dir, "parquet_info.json")
        with open(parquet_info_path, 'w') as f:
            json.dump(parquet_info, f, indent=2)

        print(f"\nDone! Created {file_index} parquet file(s) in {args.output_dir}")
        print(f"Parquet info saved to {parquet_info_path}")
        print(f"Total samples: {len(all_data)}")

    print(f"Thinking mode: {args.thinking_mode}")


if __name__ == "__main__":
    main()
