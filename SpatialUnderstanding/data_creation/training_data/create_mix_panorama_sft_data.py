#!/usr/bin/env python3
"""
Create mixed training data from panorama QA samples + SFT data.

Outputs two separate files:
  1. Panorama data → Parquet (visual reasoning with <image_start>/<image_end>)
     output_dir/panorama_visual/chunk_*.parquet

  2. SFT data (2K random samples) → JSONL (text-only reasoning with <think>)
     output_dir/sft_text_only/text_only_thinking.jsonl

Both use the same system prompt that supports both text and visual reasoning.

Usage:
    python SpatialUnderstanding/data_creation/training_data/create_mix_panorama_sft_data.py
    python SpatialUnderstanding/data_creation/training_data/create_mix_panorama_sft_data.py \
        --sft_sample_count 2000 \
        --output_dir /path/to/scratch/VisualCoT/training_data/mix_panorama_sft
"""

import argparse
import io
import json
import os
import random
import re
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


INTERLEAVED_THINK_SYSTEM_PROMPT = (
    "Let's think step by step to answer the question. "
    "For text-based thinking, enclose the process within <think> </think>, "
    "e.g. <think> thinking process here </think>. "
    "For visual thinking, enclose the content within <image_start> </image_end>, "
    "e.g. <image_start> thinking image here </image_end>. "
    "Finally conclude with the final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)


# ---------------------------------------------------------------------------
# Shared utilities (adapted from create_panorama_qa_training_data.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Part 1: Panorama samples → Parquet
# ---------------------------------------------------------------------------

def process_panorama_item(sample: dict, scene_to_sample: dict, panorama_dir: str):
    """Process a panorama sample into parquet format with visual reasoning."""
    try:
        sample_id = sample.get('sample_id', '')
        question = sample.get('question_both_views', '').strip()

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
        panorama_path = resolve_panorama_path(sample, scene_to_sample, panorama_dir)

        if not all([img1_path, img2_path, panorama_path]):
            return None

        img1_bytes = image_to_bytes(img1_path)
        img2_bytes = image_to_bytes(img2_path)
        panorama_bytes = image_to_bytes(panorama_path)

        if not all([img1_bytes, img2_bytes, panorama_bytes]):
            print(f"[WARNING] Failed to load images for sample {sample_id}")
            return None

        if options:
            options_str = "\n".join(
                [f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)]
            )
            full_question = f"{question}\n\n{options_str}"
        else:
            full_question = question

        return {
            "image_list": [img1_bytes, img2_bytes, panorama_bytes],
            "instruction_list": [INTERLEAVED_THINK_SYSTEM_PROMPT + "\n<image><image>\n" + full_question],
            "output_text_list": [
                "<image_start>",
                f"<image_end><answer>{answer_text}</answer>",
            ],
        }

    except Exception as e:
        print(f"[ERROR] Failed to process panorama sample {sample.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def write_parquet(all_data: list, output_dir: str, rows_per_group: int, groups_per_file: int):
    """Write processed samples to chunked parquet files."""
    os.makedirs(output_dir, exist_ok=True)

    schema = pa.schema([
        pa.field("image_list", pa.list_(pa.binary())),
        pa.field("instruction_list", pa.list_(pa.string())),
        pa.field("output_text_list", pa.list_(pa.string())),
    ])

    rows_per_file = rows_per_group * groups_per_file
    file_index = 0
    parquet_info = {}

    print(f"Writing parquet files to {output_dir}")
    for i in tqdm(range(0, len(all_data), rows_per_file)):
        file_data = all_data[i:i + rows_per_file]
        parquet_file = os.path.join(output_dir, f"chunk_{file_index}.parquet")
        file_index += 1

        num_row_groups = 0
        with pq.ParquetWriter(parquet_file, schema=schema, version="2.6") as writer:
            for j in range(0, len(file_data), rows_per_group):
                group_data = file_data[j:j + rows_per_group]
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

    print(f"Created {file_index} parquet file(s), info at {parquet_info_path}")
    return file_index


# ---------------------------------------------------------------------------
# Part 2: SFT samples → JSONL
# ---------------------------------------------------------------------------

def extract_thinking(response: str) -> str:
    """Extract content from <thinking>...</thinking>, dropping <spatial_thinking>."""
    match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: return full response minus any XML tags if no <thinking> found
    return re.sub(r'<[^>]+>', '', response).strip()


def process_sft_item(sample: dict, sft_image_root: str):
    """Process an SFT sample into ShareGPT JSONL format with <think> tags."""
    try:
        sample_id = str(sample.get('id', ''))
        question = sample.get('question', '').strip()
        answer = sample.get('answer', '').strip()
        response = sample.get('response', '')
        images = sample.get('images', [])

        if not question or not answer or not response:
            print(f"[WARNING] Missing fields for SFT sample {sample_id}")
            return None

        valid_images = list(images)

        # Extract only <thinking> content, drop <spatial_thinking>
        thinking_content = extract_thinking(response)

        n_images = len(valid_images)
        image_tokens = "<image>" * n_images
        human_message = f"{INTERLEAVED_THINK_SYSTEM_PROMPT}\n\n{image_tokens}\n{question}"
        gpt_response = f"<think>{thinking_content}</think><answer>{answer}</answer>"

        return {
            "conversations": [
                {"from": "human", "value": human_message},
                {"from": "gpt", "value": gpt_response},
            ],
            "image": valid_images,
            "id": sample_id,
        }

    except Exception as e:
        print(f"[ERROR] Failed to process SFT sample {sample.get('id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create mixed panorama + SFT training data"
    )
    # Panorama args
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
    # SFT args
    parser.add_argument(
        "--sft_metadata",
        default="/path/to/scratch/datasets/ViewFusion-traindata/SFT_data/metadata.jsonl",
        help="Path to SFT metadata.jsonl",
    )
    parser.add_argument(
        "--sft_image_root",
        default="/path/to/scratch/datasets/ViewFusion-traindata",
        help="Root directory for resolving SFT image relative paths",
    )
    parser.add_argument(
        "--sft_sample_count",
        type=int,
        default=2000,
        help="Number of SFT samples to randomly select",
    )
    # Output args
    parser.add_argument(
        "--output_dir",
        default="/path/to/scratch/VisualCoT/training_data/mix_panorama_sft",
        help="Output directory (will contain panorama_visual/ and sft_text_only/ subdirs)",
    )
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--rows_per_group", type=int, default=10)
    parser.add_argument("--groups_per_file", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    panorama_output_dir = os.path.join(args.output_dir, "panorama_visual")
    sft_output_dir = os.path.join(args.output_dir, "sft_text_only")

    # ------------------------------------------------------------------
    # Part 1: Panorama → Parquet
    # ------------------------------------------------------------------
    print("\n=== Part 1: Panorama data → Parquet ===")
    print(f"Loading panorama samples from {args.train_file}")
    with open(args.train_file, 'r') as f:
        panorama_samples = json.load(f)
    print(f"Loaded {len(panorama_samples)} panorama samples")

    print(f"Loading scene mapping from {args.scene_mapping_file}")
    with open(args.scene_mapping_file, 'r') as f:
        scene_to_sample = json.load(f)
    print(f"  {len(scene_to_sample)} scene mappings")

    panorama_samples = [s for s in panorama_samples if s.get('question_both_views', '').strip()]
    print(f"After filtering: {len(panorama_samples)} valid panorama samples")

    process_panorama_fn = partial(
        process_panorama_item,
        scene_to_sample=scene_to_sample,
        panorama_dir=args.panorama_dir,
    )

    print(f"Processing panorama samples with {args.max_workers} workers...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        processed = list(tqdm(
            executor.map(process_panorama_fn, panorama_samples),
            total=len(panorama_samples),
        ))
    panorama_data = [r for r in processed if r is not None]
    random.shuffle(panorama_data)
    print(f"Successfully processed {len(panorama_data)} panorama samples")

    if panorama_data:
        n_files = write_parquet(
            panorama_data, panorama_output_dir,
            args.rows_per_group, args.groups_per_file,
        )
        print(f"Panorama: {len(panorama_data)} samples → {n_files} parquet file(s)")
    else:
        print("[WARNING] No panorama samples to write.")

    # ------------------------------------------------------------------
    # Part 2: SFT → JSONL
    # ------------------------------------------------------------------
    print(f"\n=== Part 2: SFT data → JSONL (sample {args.sft_sample_count}) ===")
    print(f"Loading SFT metadata from {args.sft_metadata}")
    sft_records = []
    with open(args.sft_metadata, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                sft_records.append(json.loads(line))
    print(f"Loaded {len(sft_records)} SFT records")

    # Pre-filter to records whose images are all present, then sample
    print("Pre-filtering SFT records to those with available images...")
    available_records = [
        r for r in sft_records
        if all(os.path.exists(os.path.join(args.sft_image_root, p)) for p in r.get('images', []))
    ]
    print(f"Records with all images present: {len(available_records)} / {len(sft_records)}")

    if len(available_records) < args.sft_sample_count:
        print(f"[WARNING] Only {len(available_records)} records available, less than requested {args.sft_sample_count}")

    if len(available_records) > args.sft_sample_count:
        available_records = random.sample(available_records, args.sft_sample_count)
    sft_records = available_records
    print(f"Randomly sampled {len(sft_records)} records")

    process_sft_fn = partial(process_sft_item, sft_image_root=args.sft_image_root)

    print(f"Processing SFT samples with {args.max_workers} workers...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        processed = list(tqdm(
            executor.map(process_sft_fn, sft_records),
            total=len(sft_records),
        ))
    sft_data = [r for r in processed if r is not None]
    random.shuffle(sft_data)
    print(f"Successfully processed {len(sft_data)} SFT samples")

    if sft_data:
        os.makedirs(sft_output_dir, exist_ok=True)
        jsonl_file = os.path.join(sft_output_dir, "text_only_thinking.jsonl")
        print(f"Writing JSONL to {jsonl_file}")
        with open(jsonl_file, 'w') as f:
            for item in sft_data:
                f.write(json.dumps(item) + '\n')
        print(f"SFT: {len(sft_data)} samples → {jsonl_file}")
    else:
        print("[WARNING] No SFT samples to write.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"Panorama visual samples: {len(panorama_data)}")
    print(f"  → {panorama_output_dir}/")
    print(f"SFT text-only samples:   {len(sft_data)}")
    print(f"  → {sft_output_dir}/text_only_thinking.jsonl")
    print(f"\nSystem prompt used for both sources:")
    print(f"  {INTERLEAVED_THINK_SYSTEM_PROMPT[:80]}...")

    if panorama_data:
        s = panorama_data[0]
        print("\n--- Panorama sample preview ---")
        print(f"  instruction: {s['instruction_list'][0][:120]}...")
        print(f"  output[0]:   {s['output_text_list'][0]}")
        print(f"  output[1]:   {s['output_text_list'][1]}")
        print(f"  images:      {len(s['image_list'])} images")

    if sft_data:
        s = sft_data[0]
        print("\n--- SFT sample preview ---")
        print(f"  human:  {s['conversations'][0]['value'][:120]}...")
        print(f"  gpt:    {s['conversations'][1]['value'][:120]}...")
        print(f"  images: {s['image']}")


if __name__ == "__main__":
    main()
