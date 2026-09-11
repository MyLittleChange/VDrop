#!/usr/bin/env python3
"""
Create mixed rotation SFT training data from Infinigen + Matterport sources.

Two data sources:
  Infinigen: synthetic rotation QA from novel_qa_rotation_spatial/
  Matterport: real-world rotation QA from matterport_views/

Supports two thinking modes:
  --thinking_mode visual_only (default)
    5 images: [4 walls + panorama bridge] → Parquet files

  --thinking_mode no_thinking
    4 images: [4 walls] → JSONL (ShareGPT format)

Usage:
    python SpatialUnderstanding/data_creation/training_data/create_mix_rotation_sft_data.py
    python SpatialUnderstanding/data_creation/training_data/create_mix_rotation_sft_data.py \\
        --thinking_mode no_thinking
    python SpatialUnderstanding/data_creation/training_data/create_mix_rotation_sft_data.py \\
        --max_matterport 5000
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

# Infinigen synthetic rotation QA
INFINIGEN_ROTATION_QA_DIR = "/path/to/scratch/VisualCoT/novel_qa_rotation_spatial"

# Matterport real-world rotation QA
MATTERPORT_ROTATION_QA_DIR = "/path/to/scratch/matterport_views"

DEFAULT_OUTPUT_DIR = "/path/to/scratch/infinigen/training_data_mix_rotation"


def image_to_bytes(image_path: str) -> bytes:
    """Load image bytes. For PNG files, read raw bytes directly (no re-encode)."""
    try:
        if image_path.lower().endswith('.png'):
            with open(image_path, 'rb') as f:
                return f.read()
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            byte_stream = io.BytesIO()
            img.save(byte_stream, format='PNG')
            return byte_stream.getvalue()
    except Exception as e:
        print(f"[ERROR] Failed to load image {image_path}: {e}")
        return None


def load_infinigen_rotation_samples(rotation_qa_dir: str,
                                     n_mcq: int = 2,
                                     n_yesno: int = 1) -> list:
    """Walk Infinigen rotation QA dir and sample n_mcq MCQ + n_yesno yesno per scene.

    Dir structure: {room_part}/{scene_id}/novel_qa/rotation_qa_questions.json
    """
    if not os.path.exists(rotation_qa_dir):
        print(f"  [SKIP] Infinigen rotation dir not found: {rotation_qa_dir}")
        return []

    samples = []
    for room_part in sorted(os.listdir(rotation_qa_dir)):
        room_dir = os.path.join(rotation_qa_dir, room_part)
        if not os.path.isdir(room_dir):
            continue
        for scene_id in sorted(os.listdir(room_dir)):
            qa_file = os.path.join(room_dir, scene_id, "novel_qa", "rotation_qa_questions.json")
            if not os.path.exists(qa_file):
                continue
            try:
                with open(qa_file, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  [WARNING] Failed to load {qa_file}: {e}")
                continue

            questions = data.get("rotation_questions", [])
            mcq_qs = [q for q in questions if q.get("question_type") == "rotation_direction_mcq"]
            yesno_qs = [q for q in questions if q.get("question_type") == "rotation_proximity_yesno"]

            chosen = random.sample(mcq_qs, min(n_mcq, len(mcq_qs))) + \
                     random.sample(yesno_qs, min(n_yesno, len(yesno_qs)))

            novel_qa_dir = os.path.join(room_dir, scene_id, "novel_qa")
            for i, q in enumerate(chosen):
                samples.append({
                    "sample_id": f"infinigen_rotation_{scene_id}_{i}",
                    "question_type": q["question_type"],
                    "scene_id": scene_id,
                    "novel_qa_dir": novel_qa_dir,
                    "question": q["question"],
                    "options": q.get("options", []),
                    "correct_index": q.get("correct_index"),
                    "correct_answer": q.get("correct_answer", ""),
                    "images": q.get("images", {}),
                    "source": "infinigen",
                })
    return samples


def load_matterport_rotation_samples(rotation_qa_dir: str,
                                      n_mcq: int = 2,
                                      n_yesno: int = 1) -> list:
    """Walk Matterport rotation QA dir and sample n_mcq MCQ + n_yesno yesno per viewpoint.

    Dir structure: {scan_id}/{viewpoint_id}/rotation_qa_questions.json
    Images are in the same directory (no novel_qa/ subdirectory).
    Uses 'question_text' instead of 'question'.
    YesNo questions lack 'options'/'correct_index' — only 'correct_answer'.
    """
    if not os.path.exists(rotation_qa_dir):
        print(f"  [SKIP] Matterport rotation dir not found: {rotation_qa_dir}")
        return []

    samples = []
    for scan_id in sorted(os.listdir(rotation_qa_dir)):
        scan_dir = os.path.join(rotation_qa_dir, scan_id)
        if not os.path.isdir(scan_dir):
            continue
        for viewpoint_id in sorted(os.listdir(scan_dir)):
            vp_dir = os.path.join(scan_dir, viewpoint_id)
            qa_file = os.path.join(vp_dir, "rotation_qa_questions.json")
            if not os.path.exists(qa_file):
                continue
            try:
                with open(qa_file, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  [WARNING] Failed to load {qa_file}: {e}")
                continue

            questions = data.get("rotation_questions", [])
            mcq_qs = [q for q in questions if q.get("question_type") == "rotation_direction_mcq"]
            yesno_qs = [q for q in questions if q.get("question_type") == "rotation_proximity_yesno"]

            chosen = random.sample(mcq_qs, min(n_mcq, len(mcq_qs))) + \
                     random.sample(yesno_qs, min(n_yesno, len(yesno_qs)))

            scene_id = f"{scan_id}_{viewpoint_id}"
            for i, q in enumerate(chosen):
                samples.append({
                    "sample_id": f"matterport_rotation_{scene_id}_{i}",
                    "question_type": q["question_type"],
                    "scene_id": scene_id,
                    "novel_qa_dir": vp_dir,  # images are directly here
                    "question": q.get("question_text", q.get("question", "")),
                    "options": q.get("options", []),
                    "correct_index": q.get("correct_index"),
                    "correct_answer": q.get("correct_answer", ""),
                    "images": q.get("images", {}),
                    "source": "matterport",
                })
    return samples


def process_item_visual_only_rotation(sample: dict):
    """Visual-only thinking for rotation QA: 5 images (4 walls + panorama) → Parquet row."""
    try:
        sample_id = sample["sample_id"]
        novel_qa_dir = sample["novel_qa_dir"]
        images = sample["images"]

        correct_index = sample.get("correct_index")
        if sample["question_type"] == "rotation_direction_mcq":
            answer_text = chr(65 + correct_index) if correct_index is not None else sample["correct_answer"]
        else:
            answer_text = sample["correct_answer"]  # "Yes" or "No"

        img_keys = ["image_1", "image_2", "image_3", "image_4", "bridge_panorama"]
        img_bytes_list = []
        for key in img_keys:
            filename = images.get(key)
            if not filename:
                print(f"[WARNING] Missing image key '{key}' for {sample_id}")
                return None
            path = os.path.join(novel_qa_dir, filename)
            b = image_to_bytes(path)
            if b is None:
                return None
            img_bytes_list.append(b)

        question = sample["question"]
        full_question = question

        return {
            "image_list": img_bytes_list,
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


def process_item_no_thinking_rotation(sample: dict, image_root_dir: str):
    """No thinking for rotation QA: 4 wall images → JSONL row."""
    try:
        sample_id = sample["sample_id"]
        novel_qa_dir = sample["novel_qa_dir"]
        images = sample["images"]

        correct_index = sample.get("correct_index")
        if sample["question_type"] == "rotation_direction_mcq":
            answer_text = chr(65 + correct_index) if correct_index is not None else sample["correct_answer"]
        else:
            answer_text = sample["correct_answer"]  # "Yes" or "No"

        img_keys = ["image_1", "image_2", "image_3", "image_4"]
        img_paths = []
        for key in img_keys:
            filename = images.get(key)
            if not filename:
                print(f"[WARNING] Missing image key '{key}' for {sample_id}")
                return None
            path = os.path.join(novel_qa_dir, filename)
            if not os.path.exists(path):
                print(f"[WARNING] Image not found: {path}")
                return None
            img_paths.append(path)

        rel_paths = [os.path.relpath(p, image_root_dir) for p in img_paths]
        question = sample["question"]
        full_question = "<image><image><image><image>\n" + question
        user_message = f"{NO_THINKING_SYSTEM_PROMPT}\n\n{full_question}"
        assistant_response = f"<answer>{answer_text}</answer>"

        return {
            "conversations": [
                {"from": "human", "value": user_message},
                {"from": "gpt", "value": assistant_response},
            ],
            "image": rel_paths,
            "id": sample_id,
        }

    except Exception as e:
        print(f"[ERROR] {sample.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Create mixed rotation SFT training data from Infinigen + Matterport"
    )
    # Data sources
    parser.add_argument("--infinigen_rotation_dir", default=INFINIGEN_ROTATION_QA_DIR)
    parser.add_argument("--matterport_rotation_dir", default=MATTERPORT_ROTATION_QA_DIR)
    parser.add_argument("--no_infinigen", action="store_true", default=False,
                        help="Exclude Infinigen (synthetic) rotation samples")
    parser.add_argument("--no_matterport", action="store_true", default=False,
                        help="Exclude Matterport (real-world) rotation samples")
    parser.add_argument("--real_world_only", action="store_true", default=False,
                        help="Use only Matterport (real-world) data, equivalent to --no_infinigen")
    # Sampling
    parser.add_argument("--infinigen_n_mcq", type=int, default=2,
                        help="MCQ questions to sample per Infinigen scene")
    parser.add_argument("--infinigen_n_yesno", type=int, default=1,
                        help="YesNo questions to sample per Infinigen scene")
    parser.add_argument("--matterport_n_mcq", type=int, default=2,
                        help="MCQ questions to sample per Matterport viewpoint")
    parser.add_argument("--matterport_n_yesno", type=int, default=1,
                        help="YesNo questions to sample per Matterport viewpoint")
    parser.add_argument("--max_matterport", type=int, default=0,
                        help="Global cap on Matterport samples (0 = no cap)")
    parser.add_argument("--max_infinigen", type=int, default=0,
                        help="Global cap on Infinigen samples (0 = no cap)")
    # Output
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--thinking_mode",
        choices=["visual_only", "no_thinking"],
        default="visual_only",
    )
    parser.add_argument("--max_workers", type=int, default=32)
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

    # Load Infinigen rotation samples
    infinigen_samples = []
    if not args.no_infinigen:
        print("\nLoading Infinigen rotation samples...")
        infinigen_samples = load_infinigen_rotation_samples(
            args.infinigen_rotation_dir,
            n_mcq=args.infinigen_n_mcq,
            n_yesno=args.infinigen_n_yesno,
        )
        print(f"  Infinigen rotation: {len(infinigen_samples)} samples")
        if args.max_infinigen > 0 and len(infinigen_samples) > args.max_infinigen:
            infinigen_samples = random.sample(infinigen_samples, args.max_infinigen)
            print(f"  Capped Infinigen to {args.max_infinigen} samples")

    # Load Matterport rotation samples
    matterport_samples = []
    if not args.no_matterport:
        print("\nLoading Matterport rotation samples...")
        matterport_samples = load_matterport_rotation_samples(
            args.matterport_rotation_dir,
            n_mcq=args.matterport_n_mcq,
            n_yesno=args.matterport_n_yesno,
        )
        print(f"  Matterport rotation: {len(matterport_samples)} samples")
        if args.max_matterport > 0 and len(matterport_samples) > args.max_matterport:
            matterport_samples = random.sample(matterport_samples, args.max_matterport)
            print(f"  Capped Matterport to {args.max_matterport} samples")

    all_samples = infinigen_samples + matterport_samples
    print(f"\nTotal rotation samples: {len(all_samples)}")

    if len(all_samples) == 0:
        print("No samples to process. Exiting.")
        return

    # Category and source breakdown
    category_counts = {}
    source_counts = {}
    for s in all_samples:
        qtype = s.get("question_type", "unknown")
        source = s.get("source", "unknown")
        category_counts[qtype] = category_counts.get(qtype, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1

    print("\nSource breakdown:")
    for src, count in sorted(source_counts.items()):
        print(f"  {src}: {count}")
    print("\nCategory breakdown:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    # Select processor
    if args.thinking_mode == "no_thinking":
        process_fn = partial(process_item_no_thinking_rotation, image_root_dir=args.image_root_dir)
    else:
        process_fn = process_item_visual_only_rotation

    # Process all samples in parallel
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
