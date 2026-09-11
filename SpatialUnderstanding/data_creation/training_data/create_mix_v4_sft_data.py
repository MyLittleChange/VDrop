#!/usr/bin/env python3
"""
Create mixed SFT training data from existing panorama training samples + new V4 samples.

Existing data: /path/to/scratch/VisualCoT/training_data/panorama_train_samples_filtered.json
  (1876 samples, panoramas in rendered_panorama_train/)
New V4 data: dataset_*_filtered_V4_normalized.json
  (2025 samples, panoramas in rendered_panorama_v4/ after rendering)

Supports two thinking modes:
  --thinking_mode visual_only (default)
    3 images: [cam0, cam1, panorama] → Parquet files
    Skips samples without rendered panorama.

  --thinking_mode no_thinking
    2 images: [cam0, cam1] → JSONL (ShareGPT format)
    All samples included (no panorama needed).

Usage:
    python SpatialUnderstanding/data_creation/training_data/create_mix_v4_sft_data.py
    python SpatialUnderstanding/data_creation/training_data/create_mix_v4_sft_data.py \\
        --thinking_mode no_thinking
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

# Existing panorama training data
EXISTING_TRAIN_FILE = "/path/to/scratch/VisualCoT/training_data/panorama_train_samples_filtered.json"
EXISTING_MAPPING_FILE = "/path/to/scratch/VisualCoT/training_data/scene_to_sample_id_filtered.json"
EXISTING_PANORAMA_DIR = "/path/to/scratch/VisualCoT/infinigen/rendered_panorama_train"

# V4 training data
V4_TRAIN_FILES = [
    "/path/to/scratch/infinigen/dataset_counting_questions_filtered_V4_normalized.json",
    "/path/to/scratch/infinigen/dataset_anchor_questions_filtered_V4_normalized.json",
    "/path/to/scratch/infinigen/dataset_spatial_questions_filtered_V4_normalized.json",
    "/path/to/scratch/infinigen/dataset_relative_distance_questions_filtered_V4_normalized.json",
    "/path/to/scratch/infinigen/dataset_perspective_taking_questions_filtered_V4_normalized.json",
]
V4_MAPPING_FILE = "/path/to/scratch/infinigen/scene_to_sample_id_v4_filtered.json"
V4_PANORAMA_DIR = "/path/to/scratch/infinigen/rendered_panorama_v4"

DEFAULT_OUTPUT_DIR = "/path/to/scratch/infinigen/training_data_mix"


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


def resolve_panorama_path(sample: dict, existing_mapping: dict, v4_mapping: dict,
                           existing_panorama_dir: str, v4_panorama_dir: str) -> str:
    """Resolve panorama path: check existing mapping first, then V4 mapping."""
    scene_id = sample.get("scene_id")

    if scene_id in existing_mapping:
        first_sample_id = existing_mapping[scene_id]
        return os.path.join(existing_panorama_dir, scene_id, f"panorama_blender_limits_{first_sample_id}.png")

    if scene_id in v4_mapping:
        first_sample_id = v4_mapping[scene_id]
        return os.path.join(v4_panorama_dir, scene_id, f"panorama_blender_limits_{first_sample_id}.png")

    return None


def get_question_and_answer(sample: dict):
    """Extract question, options, and answer index from a sample."""
    question = sample.get("question_both_views", "").strip()
    options_user_1 = sample.get("options_user_1")
    options_user_2 = sample.get("options_user_2")

    if options_user_1 is not None:
        options = options_user_1
        correct_answer_idx = sample.get("user_1_gt_answer_idx")
    elif options_user_2 is not None:
        options = options_user_2
        correct_answer_idx = sample.get("user_2_gt_answer_idx")
    else:
        options = sample.get("options", [])
        correct_answer_idx = sample.get("correct_answer_idx")

    if correct_answer_idx is not None:
        answer_text = chr(65 + correct_answer_idx)
    else:
        answer_text = sample.get("correct_answer", "").strip()

    return question, options, answer_text


def process_item_visual_only(sample: dict, existing_mapping: dict, v4_mapping: dict,
                              existing_panorama_dir: str, v4_panorama_dir: str):
    """Visual-only thinking: 3 images (cam0, cam1, panorama) → Parquet row."""
    try:
        sample_id = sample.get("sample_id", "")
        question, options, answer_text = get_question_and_answer(sample)

        if not question or not answer_text:
            print(f"[WARNING] Missing question or answer for {sample_id}")
            return None

        img1_path = remap_path(sample.get("user_1_image_local_path"))
        img2_path = remap_path(sample.get("user_2_image_local_path"))
        panorama_path = resolve_panorama_path(
            sample, existing_mapping, v4_mapping, existing_panorama_dir, v4_panorama_dir
        )

        if not all([img1_path, img2_path, panorama_path]):
            return None
        if not os.path.exists(panorama_path):
            return None  # Skip if panorama not rendered yet

        img1_bytes = image_to_bytes(img1_path)
        img2_bytes = image_to_bytes(img2_path)
        panorama_bytes = image_to_bytes(panorama_path)

        if not all([img1_bytes, img2_bytes, panorama_bytes]):
            print(f"[WARNING] Failed to load images for {sample_id}")
            return None

        if options:
            options_str = "\n".join([f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)])
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
        print(f"[ERROR] {sample.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_item_no_thinking(sample: dict, image_root_dir: str):
    """No thinking: 2 images (cam0, cam1) → JSONL row."""
    try:
        sample_id = sample.get("sample_id", "")
        question, options, answer_text = get_question_and_answer(sample)

        if not question or not answer_text:
            print(f"[WARNING] Missing question or answer for {sample_id}")
            return None

        img1_path = remap_path(sample.get("user_1_image_local_path"))
        img2_path = remap_path(sample.get("user_2_image_local_path"))

        if not all([img1_path, img2_path]):
            return None

        for img_path in [img1_path, img2_path]:
            if not os.path.exists(img_path):
                print(f"[WARNING] Image not found: {img_path}")
                return None

        rel_img1 = os.path.relpath(img1_path, image_root_dir)
        rel_img2 = os.path.relpath(img2_path, image_root_dir)

        if options:
            options_str = "\n".join([f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)])
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
        print(f"[ERROR] {sample.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_samples(filepath: str) -> list:
    if not os.path.exists(filepath):
        print(f"[SKIP] Not found: {filepath}")
        return []
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} samples from {os.path.basename(filepath)}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Create mixed SFT training data from existing panorama + V4 samples"
    )
    parser.add_argument(
        "--existing_train_file", default=EXISTING_TRAIN_FILE,
    )
    parser.add_argument(
        "--existing_mapping_file", default=EXISTING_MAPPING_FILE,
    )
    parser.add_argument(
        "--existing_panorama_dir", default=EXISTING_PANORAMA_DIR,
    )
    parser.add_argument(
        "--v4_train_files", nargs="+", default=V4_TRAIN_FILES,
    )
    parser.add_argument(
        "--v4_mapping_file", default=V4_MAPPING_FILE,
    )
    parser.add_argument(
        "--v4_panorama_dir", default=V4_PANORAMA_DIR,
    )
    parser.add_argument(
        "--output_dir", default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--thinking_mode",
        choices=["visual_only", "no_thinking"],
        default="visual_only",
    )
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--rows_per_group", type=int, default=10)
    parser.add_argument("--groups_per_file", type=int, default=10)
    parser.add_argument(
        "--image_root_dir",
        default="/network/scratch",
        help="Root dir for relative image paths (no_thinking mode only)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = os.path.join(args.output_dir, args.thinking_mode)
    print(f"Thinking mode: {args.thinking_mode}")
    print(f"Output dir: {output_dir}")

    # Load existing samples
    print(f"\nLoading existing panorama training samples...")
    existing_samples = load_samples(args.existing_train_file)

    # Load V4 samples
    print(f"\nLoading V4 training samples...")
    v4_samples = []
    for filepath in args.v4_train_files:
        v4_samples.extend(load_samples(filepath))
    print(f"  Total V4: {len(v4_samples)} samples")

    all_samples = existing_samples + v4_samples
    print(f"\nTotal combined: {len(all_samples)} samples")

    # Load mappings
    existing_mapping = {}
    if os.path.exists(args.existing_mapping_file):
        with open(args.existing_mapping_file, "r") as f:
            existing_mapping = json.load(f)
        print(f"Existing mapping: {len(existing_mapping)} scenes")

    v4_mapping = {}
    if os.path.exists(args.v4_mapping_file):
        with open(args.v4_mapping_file, "r") as f:
            v4_mapping = json.load(f)
        print(f"V4 mapping: {len(v4_mapping)} scenes")

    # Filter samples with valid questions
    all_samples = [s for s in all_samples if s.get("question_both_views", "").strip()]
    print(f"After question filter: {len(all_samples)} samples")

    # Category breakdown (before processing)
    category_counts = {}
    for s in all_samples:
        qtype = s.get("question_type", "unknown")
        category_counts[qtype] = category_counts.get(qtype, 0) + 1
    print("\nCategory breakdown:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    # Select processor
    if args.thinking_mode == "no_thinking":
        process_fn = partial(process_item_no_thinking, image_root_dir=args.image_root_dir)
    else:
        process_fn = partial(
            process_item_visual_only,
            existing_mapping=existing_mapping,
            v4_mapping=v4_mapping,
            existing_panorama_dir=args.existing_panorama_dir,
            v4_panorama_dir=args.v4_panorama_dir,
        )

    # Process in parallel
    print(f"\nProcessing {len(all_samples)} samples with {args.max_workers} workers...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        processed = list(tqdm(executor.map(process_fn, all_samples), total=len(all_samples)))
    all_data = [res for res in processed if res is not None]

    random.shuffle(all_data)
    print(f"Successfully processed: {len(all_data)} samples")

    if len(all_data) == 0:
        print("No samples to write. Exiting.")
        return

    os.makedirs(output_dir, exist_ok=True)

    if args.thinking_mode == "no_thinking":
        jsonl_file = os.path.join(output_dir, "no_thinking.jsonl")
        print(f"Writing JSONL to {jsonl_file}")
        with open(jsonl_file, "w") as f:
            for item in all_data:
                f.write(json.dumps(item) + "\n")
        print(f"\nDone! {len(all_data)} samples → {jsonl_file}")

    else:
        # Remove sample_id before writing parquet
        for item in all_data:
            item.pop("sample_id", None)

        schema = pa.schema([
            pa.field("image_list", pa.list_(pa.binary())),
            pa.field("instruction_list", pa.list_(pa.string())),
            pa.field("output_text_list", pa.list_(pa.string())),
        ])

        rows_per_file = args.rows_per_group * args.groups_per_file
        file_index = 0
        parquet_info = {}

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
        with open(parquet_info_path, "w") as f:
            json.dump(parquet_info, f, indent=2)

        print(f"\nDone! {len(all_data)} samples → {file_index} parquet file(s) in {output_dir}")
        print(f"Parquet info: {parquet_info_path}")


if __name__ == "__main__":
    main()
