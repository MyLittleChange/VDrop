#!/usr/bin/env python3
"""
Create mixed SFT training data using top-down maps as the bridge image.

Reads a reference training_data.json (from the panorama-based pipeline) and
replaces panorama bridge images with top-down maps, keeping the exact same
set of training samples.

Non-rotation samples: 3 images [cam0, cam1, topdown]
Rotation samples:     5 images [wall_top, wall_right, wall_bottom, wall_left, topdown_rot_top]

Output: Parquet files with (image_list, instruction_list, output_text_list) schema,
matching the visual_only format used by BAGEL training.

Usage:
    python create_mix_topdown_sft_data.py
    python create_mix_topdown_sft_data.py --output_dir /path/to/output
"""

import argparse
import io
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


# Reference training data (panorama-based, defines which samples to include)
DEFAULT_REFERENCE_JSON = (
    "/path/to/scratch/infinigen/training_data_mix_all_rotation/"
    "training_data_mix_all_balance/syn_5_types/training_data.json"
)

# Top-down map directories (searched in order for non-rotation samples)
DEFAULT_TOPDOWN_DIRS = [
    "/path/to/scratch/infinigen/topdown_maps_v4",
    "/path/to/scratch/infinigen/topdown_maps_v5",
    "/path/to/scratch/VisualCoT/infinigen/topdown_maps_train",
]

# Rotation top-down maps
DEFAULT_ROTATION_TOPDOWN_DIR = "/path/to/scratch/VisualCoT/topdown_rotation"

# Rotation QA source (for wall images)
DEFAULT_ROTATION_QA_DIR = "/path/to/scratch/VisualCoT/novel_qa_rotation_spatial"

# Reference images dir (cam0/cam1 from json_export pipeline)
DEFAULT_REF_IMAGES_DIR = (
    "/path/to/scratch/infinigen/training_data_mix_all_rotation/"
    "training_data_mix_all_balance/syn_5_types/images"
)

# Default output
DEFAULT_OUTPUT_DIR = "/path/to/scratch/infinigen/training_data_topdown_no_rotation/visual_only"


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
    """Load image as bytes. PNG files are read raw; others re-encoded as PNG."""
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


def _split_output(output: str) -> list:
    """Split output string into output_text_list for parquet.

    Input:  '<image_start><image_end><answer>A</answer>'
    Output: ['<image_start>', '<image_end><answer>A</answer>']
    """
    marker = "<image_end>"
    idx = output.find(marker)
    if idx >= 0:
        return [output[:idx], output[idx:]]
    return [output]


def resolve_topdown_path(sample_id: str, scene_id: str, topdown_dirs: list) -> str:
    """Find topdown map for a non-rotation sample across multiple directories."""
    filename = f"topdown_agent_1_{sample_id}.png"
    for td_dir in topdown_dirs:
        path = os.path.join(td_dir, scene_id, filename)
        if os.path.exists(path):
            return path
    return None


def process_non_rotation_sample(sample: dict, topdown_dirs: list, ref_images_dir: str):
    """Process a non-rotation sample: 3 images [cam0, cam1, topdown] → Parquet row."""
    try:
        sample_id = sample["sample_id"]
        scene_id = sample["scene_id"]

        # Find topdown map
        topdown_path = resolve_topdown_path(sample_id, scene_id, topdown_dirs)
        if topdown_path is None:
            return None, "no_topdown"

        # Get cam images from reference images dir
        cam0_path = os.path.join(ref_images_dir, f"{sample_id}_cam0.png")
        cam1_path = os.path.join(ref_images_dir, f"{sample_id}_cam1.png")

        if not os.path.exists(cam0_path) or not os.path.exists(cam1_path):
            return None, "no_cam_images"

        # Load image bytes
        cam0_bytes = image_to_bytes(cam0_path)
        cam1_bytes = image_to_bytes(cam1_path)
        topdown_bytes = image_to_bytes(topdown_path)

        if not all([cam0_bytes, cam1_bytes, topdown_bytes]):
            return None, "image_load_error"

        # instruction and output are already formatted in the reference JSON
        return {
            "image_list": [cam0_bytes, cam1_bytes, topdown_bytes],
            "instruction_list": [sample["instruction"]],
            "output_text_list": _split_output(sample["output"]),
            "sample_id": sample_id,
        }, "ok"

    except Exception as e:
        print(f"[ERROR] {sample.get('sample_id', '?')}: {e}")
        return None, "error"


def process_rotation_sample(sample: dict, rot_topdown_index: dict,
                            rot_qa_index: dict, rotation_topdown_dir: str):
    """Process a rotation sample: 5 images [4 walls + topdown] → Parquet row."""
    try:
        sample_id = sample["sample_id"]
        scene_id = sample["scene_id"]

        if scene_id not in rot_topdown_index:
            return None, "no_topdown"
        if scene_id not in rot_qa_index:
            return None, "no_novel_qa"

        topdown_path = os.path.join(
            rotation_topdown_dir, rot_topdown_index[scene_id],
            scene_id, "topdown_rot_top.png"
        )
        novel_qa_dir = rot_qa_index[scene_id]

        wall_files = [
            "rotation_wall_top.png",
            "rotation_wall_right.png",
            "rotation_wall_bottom.png",
            "rotation_wall_left.png",
        ]

        img_bytes_list = []
        for filename in wall_files:
            src = os.path.join(novel_qa_dir, filename)
            if not os.path.exists(src):
                return None, "missing_wall_image"
            b = image_to_bytes(src)
            if b is None:
                return None, "image_load_error"
            img_bytes_list.append(b)

        topdown_bytes = image_to_bytes(topdown_path)
        if topdown_bytes is None:
            return None, "image_load_error"
        img_bytes_list.append(topdown_bytes)

        return {
            "image_list": img_bytes_list,
            "instruction_list": [sample["instruction"]],
            "output_text_list": _split_output(sample["output"]),
            "sample_id": sample_id,
        }, "ok"

    except Exception as e:
        print(f"[ERROR] {sample.get('sample_id', '?')}: {e}")
        return None, "error"


def main():
    parser = argparse.ArgumentParser(
        description="Create SFT training data (Parquet) with top-down maps as bridge images"
    )
    parser.add_argument("--reference_json", default=DEFAULT_REFERENCE_JSON,
                        help="Reference training_data.json (defines which samples to include)")
    parser.add_argument("--topdown_dirs", nargs="+", default=DEFAULT_TOPDOWN_DIRS,
                        help="Directories to search for non-rotation topdown maps")
    parser.add_argument("--rotation_topdown_dir", default=DEFAULT_ROTATION_TOPDOWN_DIR)
    parser.add_argument("--rotation_qa_dir", default=DEFAULT_ROTATION_QA_DIR)
    parser.add_argument("--ref_images_dir", default=DEFAULT_REF_IMAGES_DIR,
                        help="Directory containing cam0/cam1 images from json_export pipeline")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--rows_per_group", type=int, default=100,
                        help="Rows per row group in parquet files")
    parser.add_argument("--groups_per_file", type=int, default=10,
                        help="Row groups per parquet file")
    parser.add_argument("--skip_rotation", action="store_true",
                        help="Drop all rotation samples (use when rotation topdowns aren't rendered)")
    args = parser.parse_args()

    # Load reference samples
    print(f"Loading reference data from {args.reference_json}")
    with open(args.reference_json) as f:
        ref_samples = json.load(f)
    print(f"  Total reference samples: {len(ref_samples)}")

    # Split rotation vs non-rotation
    rotation_samples = [s for s in ref_samples if "rotation" in s.get("question_type", "")]
    non_rotation_samples = [s for s in ref_samples if "rotation" not in s.get("question_type", "")]
    print(f"  Non-rotation: {len(non_rotation_samples)}")
    print(f"  Rotation: {len(rotation_samples)}")

    if args.skip_rotation:
        print(f"  --skip_rotation set: dropping all {len(rotation_samples)} rotation samples")
        rotation_samples = []

    # Build rotation topdown index (scene_id -> room_part)
    print("\nIndexing rotation topdown maps...")
    rot_topdown_index = {}
    if os.path.isdir(args.rotation_topdown_dir):
        for rp in os.listdir(args.rotation_topdown_dir):
            rp_path = os.path.join(args.rotation_topdown_dir, rp)
            if not os.path.isdir(rp_path):
                continue
            for sid in os.listdir(rp_path):
                if os.path.exists(os.path.join(rp_path, sid, "topdown_rot_top.png")):
                    rot_topdown_index[sid] = rp
    print(f"  Indexed {len(rot_topdown_index)} rotation scenes with topdown maps")

    # Build rotation QA index
    rot_qa_index = {}
    if os.path.isdir(args.rotation_qa_dir):
        for rp in os.listdir(args.rotation_qa_dir):
            rp_path = os.path.join(args.rotation_qa_dir, rp)
            if not os.path.isdir(rp_path):
                continue
            for sid in os.listdir(rp_path):
                nq = os.path.join(rp_path, sid, "novel_qa")
                if os.path.isdir(nq):
                    rot_qa_index[sid] = nq

    # Process non-rotation samples
    print(f"\nProcessing {len(non_rotation_samples)} non-rotation samples...")
    results_non_rot = []
    status_counts = Counter()

    from functools import partial
    process_nr = partial(process_non_rotation_sample,
                         topdown_dirs=args.topdown_dirs,
                         ref_images_dir=args.ref_images_dir)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for result, status in tqdm(
            executor.map(process_nr, non_rotation_samples),
            total=len(non_rotation_samples),
        ):
            status_counts[status] += 1
            if result is not None:
                results_non_rot.append(result)

    print(f"  Non-rotation results: {status_counts}")

    # Process rotation samples
    print(f"\nProcessing {len(rotation_samples)} rotation samples...")
    results_rot = []
    rot_status_counts = Counter()

    process_rot = partial(process_rotation_sample,
                          rot_topdown_index=rot_topdown_index,
                          rot_qa_index=rot_qa_index,
                          rotation_topdown_dir=args.rotation_topdown_dir)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for result, status in tqdm(
            executor.map(process_rot, rotation_samples),
            total=len(rotation_samples),
        ):
            rot_status_counts[status] += 1
            if result is not None:
                results_rot.append(result)

    print(f"  Rotation results: {rot_status_counts}")

    # Combine
    all_data = results_non_rot + results_rot
    random.shuffle(all_data)

    print(f"\nTotal processed: {len(all_data)} / {len(ref_samples)}")
    print(f"  Skipped: {len(ref_samples) - len(all_data)}")

    if len(all_data) == 0:
        print("No samples to write. Exiting.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Remove sample_id before writing parquet
    for item in all_data:
        item.pop("sample_id", None)

    # Write parquet files
    schema = pa.schema([
        pa.field("image_list", pa.list_(pa.binary())),
        pa.field("instruction_list", pa.list_(pa.string())),
        pa.field("output_text_list", pa.list_(pa.string())),
    ])

    rows_per_file = args.rows_per_group * args.groups_per_file
    file_index = 0
    parquet_info = {}

    print(f"\nWriting parquet files to {args.output_dir}")
    for i in tqdm(range(0, len(all_data), rows_per_file)):
        file_data = all_data[i:i + rows_per_file]
        parquet_file = os.path.join(args.output_dir, f"chunk_{file_index}.parquet")
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

    parquet_info_path = os.path.join(args.output_dir, "parquet_info.json")
    with open(parquet_info_path, "w") as f:
        json.dump(parquet_info, f, indent=2)

    print(f"\nDone! {len(all_data)} samples → {file_index} parquet file(s) in {args.output_dir}")
    print(f"Parquet info: {parquet_info_path}")


if __name__ == "__main__":
    main()
