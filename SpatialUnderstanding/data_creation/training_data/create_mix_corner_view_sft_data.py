#!/usr/bin/env python3
"""
Create visual_only SFT training data using Round 4 photoreal Blender
**room-corner perspective views** as the bridge image.

Source of truth:
    /path/to/scratch/infinigen/training_data_mix_all_balance/
        no_thinking/no_thinking.jsonl

Round 4 corner-view lookup (one PNG per scene_id):
    /path/to/scratch/infinigen/map_questions_corner_view/
        <scene_id>/corner_view.png

Per-row layout (3 images, visual_only_think format):
    image_list:        [cam0_bytes, cam1_bytes, corner_view_bytes]
    instruction_list:  [VISUAL_ONLY_THINK_SYSTEM_PROMPT + "\n" + question]
    output_text_list:  ["<image_start>", "<image_end><answer>X</answer>"]

Usage:
    python create_mix_corner_view_sft_data.py
    python create_mix_corner_view_sft_data.py --output_dir /path/to/output
"""

import argparse
import io
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


# Source jsonl (defines which samples to include)
DEFAULT_REFERENCE_JSONL = (
    "/path/to/scratch/infinigen/training_data_mix_all_balance/"
    "no_thinking/no_thinking.jsonl"
)

# Round 4 corner-view root (one PNG per scene_id)
DEFAULT_TOPDOWN_ROOT = "/path/to/scratch/infinigen/map_questions_corner_view"
DEFAULT_BRIDGE_FILENAME = "corner_view.png"

# Cam image paths in the jsonl are relative to /network/scratch/
DEFAULT_IMAGE_ROOT = "/network/scratch"

# Default output
DEFAULT_OUTPUT_DIR = (
    "/path/to/scratch/infinigen/training_data_corner_view/visual_only"
)


VISUAL_ONLY_THINK_SYSTEM_PROMPT = (
    "Think visually to answer the question. "
    "Enclose your visual thinking within <image_start> </image_end>."
)

# Pattern: prompt is "<no-thinking system prompt>\n\n<image><image>\n<question>"
# We split off everything up to and including the last "<image>" and a
# leading newline, leaving the question (possibly with options) intact.
_IMAGE_PLACEHOLDER_RE = re.compile(r"^.*<image>\s*", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def extract_question(human_value: str) -> str:
    """Strip the system prompt + <image><image> placeholder, return the question.

    The no_thinking jsonl human field has shape:
        "<system prompt>\n\n<image><image>\n<question>\n\n<options>"
    """
    return _IMAGE_PLACEHOLDER_RE.sub("", human_value, count=1).strip()


def extract_answer(gpt_value: str) -> str:
    """Extract the letter / text inside <answer>...</answer>."""
    m = _ANSWER_RE.search(gpt_value or "")
    return m.group(1).strip() if m else ""


def scene_id_from_image_path(rel_path: str):
    """Image path is e.g. q/USERNAME/infinigen/spatial/<rp>/<scene_id>/frames/...

    Scene id is the path segment immediately before "frames".
    """
    parts = rel_path.split("/")
    if "frames" in parts:
        i = parts.index("frames")
        if i >= 1:
            return parts[i - 1]
    return None


def image_to_bytes(image_path: str):
    """Load image as bytes. PNG files read raw; others re-encoded as PNG."""
    try:
        if image_path.lower().endswith(".png"):
            with open(image_path, "rb") as f:
                return f.read()
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception as e:
        print(f"[ERROR] Failed to load image {image_path}: {e}")
        return None


def process_row(row: dict, topdown_root: str, image_root: str,
                bridge_filename: str = DEFAULT_BRIDGE_FILENAME):
    """Build a single parquet row from one no_thinking.jsonl entry.

    Returns (row_dict | None, status_tag).
    """
    try:
        sample_id = row.get("id", "")
        images = row.get("image", [])
        convs = row.get("conversations", [])
        if len(images) < 2:
            return None, "fewer_than_2_images"

        cam0_rel, cam1_rel = images[0], images[1]
        scene_id = scene_id_from_image_path(cam0_rel)
        if not scene_id:
            return None, "no_scene_id"

        topdown_path = os.path.join(topdown_root, scene_id, bridge_filename)
        if not os.path.exists(topdown_path):
            return None, "no_bridge"

        cam0_path = os.path.join(image_root, cam0_rel)
        cam1_path = os.path.join(image_root, cam1_rel)
        if not (os.path.exists(cam0_path) and os.path.exists(cam1_path)):
            return None, "no_cam_images"

        human = next((c["value"] for c in convs if c.get("from") == "human"), "")
        gpt = next((c["value"] for c in convs if c.get("from") == "gpt"), "")

        question = extract_question(human)
        answer = extract_answer(gpt)
        if not question or not answer:
            return None, "no_qa"

        cam0_bytes = image_to_bytes(cam0_path)
        cam1_bytes = image_to_bytes(cam1_path)
        topdown_bytes = image_to_bytes(topdown_path)
        if not all([cam0_bytes, cam1_bytes, topdown_bytes]):
            return None, "image_load_error"

        return {
            "image_list": [cam0_bytes, cam1_bytes, topdown_bytes],
            "instruction_list": [
                VISUAL_ONLY_THINK_SYSTEM_PROMPT + "\n" + question
            ],
            "output_text_list": [
                "<image_start>",
                f"<image_end><answer>{answer}</answer>",
            ],
            "sample_id": sample_id,
        }, "ok"

    except Exception as e:
        print(f"[ERROR] {row.get('id', '?')}: {e}")
        return None, "error"


def main():
    parser = argparse.ArgumentParser(
        description="Create visual_only SFT parquet data with Round 3 topdown bridges"
    )
    parser.add_argument("--reference_jsonl", default=DEFAULT_REFERENCE_JSONL,
                        help="Source jsonl that defines the canonical sample set.")
    parser.add_argument("--topdown_root", default=DEFAULT_TOPDOWN_ROOT,
                        help="Bridge-image root: <root>/<scene_id>/<bridge_filename>")
    parser.add_argument("--bridge_filename", default=DEFAULT_BRIDGE_FILENAME,
                        help="Filename of the per-scene bridge PNG.")
    parser.add_argument("--image_root", default=DEFAULT_IMAGE_ROOT,
                        help="Prefix for image[] paths in the jsonl.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--rows_per_group", type=int, default=100)
    parser.add_argument("--groups_per_file", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: only process the first N samples (smoke test).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.reference_jsonl}")
    with open(args.reference_jsonl) as f:
        rows = [json.loads(ln) for ln in f]
    print(f"  Total samples: {len(rows)}")

    if args.limit:
        rows = rows[: args.limit]
        print(f"  Limiting to first {len(rows)}")

    # Process
    print(f"\nProcessing {len(rows)} samples...")
    results = []
    status_counts = Counter()
    fn = partial(process_row,
                 topdown_root=args.topdown_root,
                 image_root=args.image_root,
                 bridge_filename=args.bridge_filename)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for result, status in tqdm(executor.map(fn, rows), total=len(rows)):
            status_counts[status] += 1
            if result is not None:
                results.append(result)

    print(f"\nStatus counts: {dict(status_counts)}")
    print(f"Built {len(results)} parquet rows")

    if not results:
        print("No samples to write. Exiting.")
        return

    random.Random(args.seed).shuffle(results)

    os.makedirs(args.output_dir, exist_ok=True)

    # Strip sample_id before writing
    for item in results:
        item.pop("sample_id", None)

    schema = pa.schema([
        pa.field("image_list", pa.list_(pa.binary())),
        pa.field("instruction_list", pa.list_(pa.string())),
        pa.field("output_text_list", pa.list_(pa.string())),
    ])

    rows_per_file = args.rows_per_group * args.groups_per_file
    file_index = 0
    parquet_info = {}

    print(f"\nWriting parquet files to {args.output_dir}")
    for i in tqdm(range(0, len(results), rows_per_file)):
        chunk = results[i:i + rows_per_file]
        out_path = os.path.join(args.output_dir, f"chunk_{file_index}.parquet")
        file_index += 1
        n_groups = 0
        with pq.ParquetWriter(out_path, schema=schema, version="2.6") as writer:
            for j in range(0, len(chunk), args.rows_per_group):
                grp = chunk[j:j + args.rows_per_group]
                tbl = pa.Table.from_pylist(grp, schema=schema)
                writer.write_table(tbl)
                n_groups += 1
        parquet_info[out_path] = {"num_row_groups": n_groups, "num_rows": len(chunk)}

    info_path = os.path.join(args.output_dir, "parquet_info.json")
    with open(info_path, "w") as f:
        json.dump(parquet_info, f, indent=2)

    print(f"\nDone. {len(results)} samples → {file_index} parquet file(s) in {args.output_dir}")
    print(f"Parquet info: {info_path}")


if __name__ == "__main__":
    main()
