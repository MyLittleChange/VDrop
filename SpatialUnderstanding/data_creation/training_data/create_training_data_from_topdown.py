#!/usr/bin/env python3
"""
Create training data from Gemini topdown inference results.

Input: Inference results JSON with normalized image paths and questions.
Output: Training data in parquet or JSONL format depending on thinking mode.

Supports two thinking modes:
  --thinking_mode visual_only (default)
    Generated reasoning image only (no text reasoning).
    Images: [user_1_image, user_2_image, topdown_map]
    Output: ["<image_start>", "<image_end><answer>LETTER</answer>"]
    Format: Parquet files

  --thinking_mode no_thinking
    Direct answer without any reasoning.
    Images: [user_1_image, user_2_image]
    Output: ShareGPT JSONL with <answer>LETTER</answer>

Usage:
    python scripts/create_training_data_from_topdown.py
    python scripts/create_training_data_from_topdown.py --thinking_mode no_thinking
    python scripts/create_training_data_from_topdown.py --input_file /path/to/results.json
"""

import argparse
import io
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


VISUAL_ONLY_THINK_SYSTEM_PROMPT = (
    "Think visually to answer the question. "
    "Enclose your visual thinking within <image_start> </image_end>."
)

NO_THINKING_SYSTEM_PROMPT = (
    "Answer the question directly. "
    "Provide your final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)


def image_to_bytes(image_path: str) -> bytes:
    """Load image from path and convert to PNG bytes."""
    try:
        with Image.open(image_path) as img:
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
        "/path/to/scratch",
    )
    path = path.replace(
        "/path/to/scratch",
        "/path/to/scratch",
    )
    return path


def normalize_relative_sample(sample: dict) -> dict:
    """Normalize a relative_dataset sample to match Gemini format fields."""
    # Determine which agent has the question
    if sample.get("user_1_question"):
        options = sample.get("options_user_1", [])
        gt_idx = sample.get("user_1_gt_answer_idx")
    else:
        options = sample.get("options_user_2", [])
        gt_idx = sample.get("user_2_gt_answer_idx")

    return {
        "sample_id": sample.get("sample_id", ""),
        "question": sample.get("question_both_views", ""),
        "options": options,
        "correct_answer_idx": gt_idx,
        "user_1_image_path": sample.get("user_1_image_local_path"),
        "user_2_image_path": sample.get("user_2_image_local_path"),
        "topdown_path": sample.get("topdown_path"),
    }


def process_item(result: dict):
    """Process a single inference result into training format."""
    try:
        sample_id = result.get("sample_id", "")
        question = result.get("question", "")
        options = result.get("options", [])
        correct_answer_idx = result.get("correct_answer_idx")

        if not question or correct_answer_idx is None:
            print(f"[WARNING] Missing question or answer for sample {sample_id}")
            return None

        # Image paths (already normalized in inference results, but remap just in case)
        user_1_path = remap_path(result.get("user_1_image_path"))
        user_2_path = remap_path(result.get("user_2_image_path"))
        topdown_path = remap_path(result.get("topdown_path"))

        if not all([user_1_path, user_2_path, topdown_path]):
            print(f"[WARNING] Missing image paths for sample {sample_id}")
            return None

        # Load images
        img1_bytes = image_to_bytes(user_1_path)
        img2_bytes = image_to_bytes(user_2_path)
        topdown_bytes = image_to_bytes(topdown_path)

        if not all([img1_bytes, img2_bytes, topdown_bytes]):
            print(f"[WARNING] Failed to load images for sample {sample_id}")
            return None

        # Format question with options
        options_str = "\n".join(
            [f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)]
        )
        full_question = f"{question}\n\n{options_str}"

        # Answer letter
        answer_letter = chr(65 + correct_answer_idx)

        return {
            "image_list": [img1_bytes, img2_bytes, topdown_bytes],
            "instruction_list": [VISUAL_ONLY_THINK_SYSTEM_PROMPT + "\n" + full_question],
            "output_text_list": [
                "<image_start>",
                f"<image_end><answer>{answer_letter}</answer>",
            ],
            "sample_id": sample_id,
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {result.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_item_no_thinking(result: dict, image_root_dir: str = "/network/scratch"):
    """No thinking: direct answer without any reasoning, ShareGPT/JSONL format."""
    try:
        sample_id = result.get("sample_id", "")
        question = result.get("question", "")
        options = result.get("options", [])
        correct_answer_idx = result.get("correct_answer_idx")

        if not question or correct_answer_idx is None:
            print(f"[WARNING] Missing question or answer for sample {sample_id}")
            return None

        user_1_path = remap_path(result.get("user_1_image_path"))
        user_2_path = remap_path(result.get("user_2_image_path"))

        if not all([user_1_path, user_2_path]):
            print(f"[WARNING] Missing image paths for sample {sample_id}")
            return None

        # Verify images exist
        for img_path in [user_1_path, user_2_path]:
            if not os.path.exists(img_path):
                print(f"[WARNING] Image not found: {img_path}")
                return None

        # Convert absolute paths to relative paths w.r.t. image_root_dir
        rel_img_0 = os.path.relpath(user_1_path, image_root_dir)
        rel_img_1 = os.path.relpath(user_2_path, image_root_dir)

        # Format question with options
        options_str = "\n".join(
            [f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)]
        )
        full_question = f"{question}\n\n{options_str}"

        answer_letter = chr(65 + correct_answer_idx)

        # Build ShareGPT format: direct answer, no thinking
        user_message = f"{NO_THINKING_SYSTEM_PROMPT}\n\n<image><image>\n{full_question}"
        assistant_response = f"<answer>{answer_letter}</answer>"

        return {
            "conversations": [
                {"from": "human", "value": user_message},
                {"from": "gpt", "value": assistant_response},
            ],
            "image": [rel_img_0, rel_img_1],
            "id": sample_id,
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {result.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Create training data from topdown inference results"
    )
    parser.add_argument(
        "--input_file",
        default="/path/to/scratch/VisualCoT/Gemini_topdown/gemini_3_pro_preview_topdown_high_train.json",
        help="Path to inference results JSON",
    )
    parser.add_argument(
        "--output_dir",
        default="/path/to/scratch/VisualCoT/training_data/spatial_rel_topdown_visual_only_train",
        help="Output directory for parquet/JSONL files",
    )
    parser.add_argument(
        "--thinking_mode",
        choices=["visual_only", "no_thinking"],
        default="visual_only",
        help=(
            "Thinking mode for training data: "
            "visual_only = reasoning image only (default), "
            "no_thinking = direct answer without any reasoning"
        ),
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of parallel workers for processing",
    )
    parser.add_argument(
        "--rows_per_group",
        type=int,
        default=10,
        help="Rows per row group in parquet files",
    )
    parser.add_argument(
        "--groups_per_file",
        type=int,
        default=10,
        help="Row groups per parquet file",
    )
    parser.add_argument(
        "--image_root_dir",
        default="/path/to/scratch/spatial_collab_dataset/scenes",
        help=(
            "Root directory for images (no_thinking mode only). "
            "Image paths in JSONL will be relative to this directory. "
            "Must match data_dir in dataset_info.py."
        ),
    )
    parser.add_argument(
        "--input_file_2",
        default='/path/to/scratch/VisualCoT/spatial_collab_dataset/relative_dataset_V_Final_2000_train.json',
        help="Optional second input file (relative dataset format, top-level JSON list)",
    )

    args = parser.parse_args()

    # Select processor based on thinking mode
    if args.thinking_mode == "no_thinking":
        from functools import partial
        process_fn = partial(process_item_no_thinking, image_root_dir=args.image_root_dir)
    else:
        process_fn = process_item
    print(f"Thinking mode: {args.thinking_mode}")

    # Load inference results
    print(f"Loading inference results from {args.input_file}...")
    with open(args.input_file, "r") as f:
        data = json.load(f)

    # Support both {"results": [...]} and top-level list formats
    results = data.get("results", data) if isinstance(data, dict) else data
    print(f"Loaded {len(results)} results from primary file")

    # Load optional second dataset (relative format)
    if args.input_file_2:
        print(f"Loading relative dataset from {args.input_file_2}...")
        with open(args.input_file_2, "r") as f:
            rel_data = json.load(f)
        rel_results = [normalize_relative_sample(s) for s in rel_data]
        print(f"Loaded {len(rel_results)} relative samples")
        results.extend(rel_results)
        print(f"Total combined: {len(results)} results")

    # Filter out results without required fields
    required_fields_filter = [
        lambda r: r.get("question", "").strip(),
        lambda r: r.get("correct_answer_idx") is not None,
        lambda r: r.get("user_1_image_path"),
        lambda r: r.get("user_2_image_path"),
    ]
    if args.thinking_mode == "visual_only":
        required_fields_filter.append(lambda r: r.get("topdown_path"))

    results = [
        r for r in results
        if all(f(r) for f in required_fields_filter)
    ]
    print(f"After filtering: {len(results)} valid results")

    # Process items in parallel
    print(f"Processing samples with {args.max_workers} workers...")
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

    if args.thinking_mode == "no_thinking":
        # Write JSONL for vlm_sft dataset (ShareGPT format)
        jsonl_file = os.path.join(args.output_dir, "no_thinking.jsonl")
        print(f"Writing JSONL to {jsonl_file}")
        with open(jsonl_file, "w") as f:
            for item in all_data:
                f.write(json.dumps(item) + "\n")

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
        # Remove sample_id before writing parquet
        for item in all_data:
            if "sample_id" in item:
                del item["sample_id"]

        # Write to parquet files
        rows_per_file = args.rows_per_group * args.groups_per_file
        file_index = 0
        parquet_info = {}

        schema = pa.schema([
            pa.field("image_list", pa.list_(pa.binary())),
            pa.field("instruction_list", pa.list_(pa.string())),
            pa.field("output_text_list", pa.list_(pa.string())),
        ])

        print(f"Writing parquet files to {args.output_dir}")
        for i in tqdm(range(0, len(all_data), rows_per_file)):
            file_data = all_data[i : i + rows_per_file]
            parquet_file = os.path.join(args.output_dir, f"chunk_{file_index}.parquet")
            file_index += 1

            num_row_groups = 0
            with pq.ParquetWriter(parquet_file, schema=schema, version="2.6") as writer:
                for j in range(0, len(file_data), args.rows_per_group):
                    group_data = file_data[j : j + args.rows_per_group]
                    group_table = pa.Table.from_pylist(group_data, schema=schema)
                    writer.write_table(group_table)
                    num_row_groups += 1

            parquet_info[parquet_file] = {
                "num_row_groups": num_row_groups,
                "num_rows": len(file_data),
            }

        # Write parquet_info.json
        parquet_info_path = os.path.join(args.output_dir, "parquet_info.json")
        with open(parquet_info_path, "w") as f:
            json.dump(parquet_info, f, indent=2)

        print(f"\nDone! Created {file_index} parquet file(s) in {args.output_dir}")
        print(f"Parquet info saved to {parquet_info_path}")
        print(f"Total samples: {len(all_data)}")

    print(f"Thinking mode: {args.thinking_mode}")


if __name__ == "__main__":
    main()
