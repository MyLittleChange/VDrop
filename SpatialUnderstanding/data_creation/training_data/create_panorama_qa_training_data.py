#!/usr/bin/env python3
"""
Create training data from panorama QA samples.

Input: Two perspective images + question, with rendered panorama as
the visual reasoning image.

Supports two thinking modes:
  --thinking_mode visual_only (default)
    Generated reasoning image only (no text reasoning).
    Images: [user_1_image, user_2_image, panorama]
    Output: ["<image_start>", "<image_end><answer>LETTER</answer>"]
    Format: Parquet files

  --thinking_mode no_thinking
    Direct answer without any reasoning.
    Images: [user_1_image, user_2_image]
    Output: ShareGPT JSONL with <answer>LETTER</answer>

Usage:
    python SpatialUnderstanding/data_creation/training_data/create_panorama_qa_training_data.py
    python SpatialUnderstanding/data_creation/training_data/create_panorama_qa_training_data.py --thinking_mode no_thinking
"""

import argparse
import io
import json
import os
import random
from functools import partial

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


def resolve_panorama_path(sample: dict, scene_to_sample: dict, panorama_dir: str) -> str:
    """Resolve the rendered panorama path for a sample using scene-to-sample mapping."""
    scene_id = sample.get('scene_id')
    first_sample_id = scene_to_sample.get(scene_id)
    if not first_sample_id:
        return None
    return os.path.join(panorama_dir, scene_id, f"panorama_blender_limits_{first_sample_id}.png")


def process_item(sample: dict, scene_to_sample: dict, panorama_dir: str):
    """Visual-only thinking: reasoning image only, no text reasoning."""
    try:
        sample_id = sample.get('sample_id', '')
        question = sample.get('question_both_views', '').strip()

        # Prefer user-specific options/answer if available
        options_user_1 = sample.get('options_user_1')
        options_user_2 = sample.get('options_user_2')
        if options_user_1 is not None:
            options = options_user_1
            correct_answer_idx = sample.get('user_1_gt_answer_idx')
        elif options_user_2 is not None:
            options = options_user_2
            correct_answer_idx = sample.get('user_2_gt_answer_idx')
        else:
            options = sample.get('options', [])
            correct_answer_idx = sample.get('correct_answer_idx')

        if not question or correct_answer_idx is None:
            # Try correct_answer as fallback for datasets without options
            correct_answer = sample.get('correct_answer', '').strip()
            if not question or not correct_answer:
                print(f"[WARNING] Missing question or answer for sample {sample_id}")
                return None
            # Use correct_answer directly as answer_text
            answer_text = correct_answer
        else:
            answer_text = chr(65 + correct_answer_idx)

        img1_path = remap_path(sample.get('user_1_image_local_path'))
        img2_path = remap_path(sample.get('user_2_image_local_path'))
        panorama_path = resolve_panorama_path(sample, scene_to_sample, panorama_dir)

        if not all([img1_path, img2_path, panorama_path]):
            return None

        img1_bytes = image_to_bytes(img1_path)
        img2_bytes = image_to_bytes(img2_path)
        panorama_bytes = image_to_bytes(panorama_path)

        if not all([img1_bytes, img2_bytes, panorama_bytes]):
            print(f"[WARNING] Failed to load images for sample {sample_id}")
            return None

        # Format question with options if available
        if options:
            options_str = "\n".join(
                [f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)]
            )
            full_question = f"{question}\n\n{options_str}"
        else:
            full_question = question

        return {
            "image_list": [img1_bytes, img2_bytes, panorama_bytes],
            "instruction_list": [VISUAL_ONLY_THINK_SYSTEM_PROMPT + "\n" + full_question],
            "output_text_list": [
                "<image_start>",
                f"<image_end><answer>{answer_text}</answer>",
            ],
            "sample_id": sample_id,
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {sample.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_item_no_thinking(sample: dict, image_root_dir: str):
    """No thinking: direct answer without any reasoning, ShareGPT/JSONL format."""
    try:
        sample_id = sample.get('sample_id', '')
        question = sample.get('question_both_views', '').strip()

        # Prefer user-specific options/answer if available
        options_user_1 = sample.get('options_user_1')
        options_user_2 = sample.get('options_user_2')
        if options_user_1 is not None:
            options = options_user_1
            correct_answer_idx = sample.get('user_1_gt_answer_idx')
        elif options_user_2 is not None:
            options = options_user_2
            correct_answer_idx = sample.get('user_2_gt_answer_idx')
        else:
            options = sample.get('options', [])
            correct_answer_idx = sample.get('correct_answer_idx')

        if not question or correct_answer_idx is None:
            correct_answer = sample.get('correct_answer', '').strip()
            if not question or not correct_answer:
                print(f"[WARNING] Missing question or answer for sample {sample_id}")
                return None
            answer_text = correct_answer
        else:
            answer_text = chr(65 + correct_answer_idx)

        img1_path = remap_path(sample.get('user_1_image_local_path'))
        img2_path = remap_path(sample.get('user_2_image_local_path'))

        if not all([img1_path, img2_path]):
            return None

        for img_path in [img1_path, img2_path]:
            if not os.path.exists(img_path):
                print(f"[WARNING] Image not found: {img_path}")
                return None

        rel_img1 = os.path.relpath(img1_path, image_root_dir)
        rel_img2 = os.path.relpath(img2_path, image_root_dir)

        # Format question with options if available
        if options:
            options_str = "\n".join(
                [f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)]
            )
            full_question = f"{question}\n\n{options_str}"
        else:
            full_question = question

        user_message = f"{NO_THINKING_SYSTEM_PROMPT}\n\n<image><image>\n{full_question}"
        assistant_response = f"<answer>{answer_text}</answer>"

        return {
            "conversations": [
                {"from": "human", "value": user_message},
                {"from": "gpt", "value": assistant_response},
            ],
            "image": [rel_img1, rel_img2],
            "id": sample_id,
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {sample.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Create training data from panorama QA samples"
    )
    parser.add_argument(
        "--train_file",
        default="/path/to/scratch/VisualCoT/training_data/panorama_train_samples_filtered.json",
        help="Path to panorama_train_samples.json",
    )
    parser.add_argument(
        "--scene_mapping_file",
        default="/path/to/scratch/VisualCoT/training_data/scene_to_sample_id_filtered.json",
        help="Path to scene_to_sample_id.json",
    )
    parser.add_argument(
        "--panorama_dir",
        default="/path/to/scratch/VisualCoT/infinigen/rendered_panorama_train",
        help="Directory containing rendered panoramas (<scene_id>/panorama_blender_limits_*.png)",
    )
    parser.add_argument(
        "--output_dir",
        default="/path/to/scratch/VisualCoT/training_data/panorama_qa",
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
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--rows_per_group", type=int, default=10)
    parser.add_argument("--groups_per_file", type=int, default=10)
    parser.add_argument(
        "--image_root_dir",
        default="/path/to/scratch/spatial_collab_dataset/scenes",
        help=(
            "Root directory for images (no_thinking mode only). "
            "Image paths in JSONL will be relative to this directory."
        ),
    )

    args = parser.parse_args()

    # Append thinking mode to output dir
    output_dir = os.path.join(args.output_dir, args.thinking_mode)
    print(f"Thinking mode: {args.thinking_mode}")
    print(f"Output dir: {output_dir}")

    # Load train samples
    print(f"Loading train samples from {args.train_file}")
    with open(args.train_file, 'r') as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples")

    # Load scene-to-sample mapping (needed for panorama path resolution)
    scene_to_sample = {}
    if args.thinking_mode == "visual_only":
        print(f"Loading scene mapping from {args.scene_mapping_file}")
        with open(args.scene_mapping_file, 'r') as f:
            scene_to_sample = json.load(f)
        print(f"  {len(scene_to_sample)} scene mappings")

    # Select processor based on thinking mode
    if args.thinking_mode == "no_thinking":
        process_fn = partial(process_item_no_thinking, image_root_dir=args.image_root_dir)
    else:
        process_fn = partial(process_item, scene_to_sample=scene_to_sample, panorama_dir=args.panorama_dir)
    print(f"Thinking mode: {args.thinking_mode}")

    # Filter samples with valid questions
    samples = [s for s in samples if s.get('question_both_views', '').strip()]
    print(f"After filtering: {len(samples)} valid samples")

    # Process in parallel
    print(f"Processing samples with {args.max_workers} workers...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        processed = list(tqdm(executor.map(process_fn, samples), total=len(samples)))
        all_data = [res for res in processed if res is not None]

    random.shuffle(all_data)
    print(f"Successfully processed {len(all_data)} samples")

    if len(all_data) == 0:
        print("No samples to write. Exiting.")
        return

    os.makedirs(output_dir, exist_ok=True)

    if args.thinking_mode == "no_thinking":
        # Write JSONL (ShareGPT format)
        jsonl_file = os.path.join(output_dir, "no_thinking.jsonl")
        print(f"Writing JSONL to {jsonl_file}")
        with open(jsonl_file, 'w') as f:
            for item in all_data:
                f.write(json.dumps(item) + '\n')

        print(f"\nDone! Created {jsonl_file}")
        print(f"Total samples: {len(all_data)}")

        sample = all_data[0]
        print("\n--- Sample entry ---")
        print(f"ID: {sample['id']}")
        print(f"Images: {sample['image']}")
        print(f"Human: {sample['conversations'][0]['value'][:200]}...")
        print(f"GPT: {sample['conversations'][1]['value'][:200]}...")
    else:
        # Remove sample_id before writing parquet
        for item in all_data:
            if 'sample_id' in item:
                del item['sample_id']

        # Write to parquet files
        rows_per_file = args.rows_per_group * args.groups_per_file
        file_index = 0
        parquet_info = {}

        schema = pa.schema([
            pa.field("image_list", pa.list_(pa.binary())),
            pa.field("instruction_list", pa.list_(pa.string())),
            pa.field("output_text_list", pa.list_(pa.string())),
        ])

        print(f"Writing parquet files to {output_dir}")
        for i in tqdm(range(0, len(all_data), rows_per_file)):
            file_data = all_data[i:i + rows_per_file]
            parquet_file = os.path.join(output_dir, f"chunk_{file_index}.parquet")
            file_index += 1

            num_row_groups = 0
            with pq.ParquetWriter(parquet_file, schema=schema, version="2.6") as writer:
                for j in range(0, len(file_data), args.rows_per_group):
                    group_data = file_data[j:j + args.rows_per_group]
                    group_table = pa.Table.from_pylist(group_data, schema=schema)
                    writer.write_table(group_table)
                    num_row_groups += 1

            parquet_info[parquet_file] = {
                "num_row_groups": num_row_groups,
                "num_rows": len(file_data),
            }

        parquet_info_path = os.path.join(output_dir, "parquet_info.json")
        with open(parquet_info_path, 'w') as f:
            json.dump(parquet_info, f, indent=2)

        print(f"\nDone! Created {file_index} parquet file(s) in {output_dir}")
        print(f"Parquet info saved to {parquet_info_path}")
        print(f"Total samples: {len(all_data)}")

    print(f"Thinking mode: {args.thinking_mode}")


if __name__ == "__main__":
    main()
