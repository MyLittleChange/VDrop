#!/usr/bin/env python3
"""
Create mixed SFT training data using point-matching (PM) annotations as the
bridge image, analogous to create_mix_topdown_sft_data.py.

Each PM annotation provides cross-view shared-object markers (same color =
same object in both cam frames). The bridge here is a side-by-side composite
of annotated_cam0.png + annotated_cam1.png: one image that tells the model
"object X in view 0 = object X in view 1", serving the same role panorama or
topdown does in the other recipes.

Per-sample layout: 3 images [cam0, cam1, pm_bridge]
Output prompt:    visual_only style — `<image_start><image_end><answer>X</answer>`

Reads samples from no_thinking.jsonl (the canonical reference for which
samples to include) and skips rotation entries.

Usage:
    python create_mix_pm_sft_data.py
    python create_mix_pm_sft_data.py --output_dir /path/to/output
"""
import argparse
import io
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


VISUAL_ONLY_THINK_SYSTEM_PROMPT = (
    "Think visually to answer the question. "
    "Enclose your visual thinking within <image_start> </image_end>."
)

DEFAULT_REFERENCE_JSONL = (
    "/path/to/scratch/infinigen/training_data_mix_all_balance/"
    "no_thinking/no_thinking.jsonl"
)

# PM annotation roots (one per source).
PM_ANNO_ROOTS = {
    "v4":    "/path/to/scratch/anno_full/v4",
    "ankur": "/path/to/scratch/anno_full/ankur",
    "v5":    "/path/to/scratch/VisualCoT/training_data/map_questions_synth/annotation_v5",
}

DEFAULT_OUTPUT_DIR = "/path/to/scratch/infinigen/training_data_pm_no_rotation/visual_only"

DEFAULT_IMAGE_PREFIX = "/network/scratch"


def remap_path(path: str) -> str:
    """Remap home-dir paths to scratch."""
    if path is None:
        return None
    path = path.replace("/path/to/scratch", "/path/to/scratch")
    path = path.replace("/path/to/scratch", "/path/to/scratch")
    return path


def detect_source(image_path: str) -> str:
    if "/infinigen/outputs_rendered/" in image_path:
        return "v4"
    if "/infinigen/spatial/" in image_path:
        return "v5"
    if "infinigen_debang" in image_path or "v00_filtered" in image_path or "ankur" in image_path:
        return "ankur"
    return None


def extract_scene_id(image_path: str) -> str:
    parts = image_path.split("/")
    try:
        i = parts.index("frames")
        return parts[i - 1]
    except ValueError:
        return None


def load_image_bytes_png(path: str) -> bytes:
    """Read PNG raw, otherwise re-encode to PNG."""
    try:
        if path.lower().endswith(".png"):
            with open(path, "rb") as f:
                return f.read()
        with Image.open(path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return None


def make_pm_bridge_bytes(annotated_cam0: str, annotated_cam1: str) -> bytes:
    """Side-by-side composite of the two annotated views as the PM bridge."""
    try:
        with Image.open(annotated_cam0) as i0, Image.open(annotated_cam1) as i1:
            if i0.mode != "RGB":
                i0 = i0.convert("RGB")
            if i1.mode != "RGB":
                i1 = i1.convert("RGB")
            h = max(i0.height, i1.height)
            w = i0.width + i1.width
            canvas = Image.new("RGB", (w, h), (0, 0, 0))
            canvas.paste(i0, (0, 0))
            canvas.paste(i1, (i0.width, 0))
            buf = io.BytesIO()
            canvas.save(buf, format="PNG")
            return buf.getvalue()
    except Exception as e:
        print(f"[ERROR] Failed to build bridge from {annotated_cam0} + {annotated_cam1}: {e}")
        return None


def parse_no_thinking_sample(line: str, image_prefix: str):
    """Parse one no_thinking.jsonl line into (sample_id, source, scene_id,
    cam0_path, cam1_path, question, answer). Returns None if any required
    field is missing or the sample is rotation."""
    d = json.loads(line)
    sample_id = d.get("id")
    if not sample_id or "rotation" in sample_id:
        return None
    images = d.get("image") or []
    if len(images) < 2:
        return None
    convs = d.get("conversations") or []
    human = next((c["value"] for c in convs if c.get("from") == "human"), None)
    gpt   = next((c["value"] for c in convs if c.get("from") == "gpt"),   None)
    if human is None or gpt is None:
        return None
    img0 = remap_path(images[0])
    img1 = remap_path(images[1])
    if not img0.startswith("/"):
        img0 = os.path.join(image_prefix, img0)
    if not img1.startswith("/"):
        img1 = os.path.join(image_prefix, img1)
    src = detect_source(img0)
    sid = extract_scene_id(img0)
    if not src or not sid:
        return None
    # Recover the question portion (after the trailing `<image><image>\n`).
    last_img = human.rfind("<image>")
    question = human[last_img + len("<image>"):].lstrip("\n").strip() if last_img != -1 else human.strip()
    return {
        "sample_id": sample_id,
        "source":    src,
        "scene_id":  sid,
        "cam0":      img0,
        "cam1":      img1,
        "question":  question,
        "gpt":       gpt,
    }


def process_sample(sample: dict, pm_anno_roots: dict):
    """Build a parquet row: [cam0_bytes, cam1_bytes, pm_bridge_bytes]."""
    try:
        anno_root = pm_anno_roots.get(sample["source"])
        if anno_root is None:
            return None, "unknown_source"
        anno_dir = os.path.join(anno_root, sample["scene_id"])
        a0 = os.path.join(anno_dir, "annotated_cam0.png")
        a1 = os.path.join(anno_dir, "annotated_cam1.png")
        if not (os.path.exists(a0) and os.path.exists(a1)):
            return None, "no_pm_annotation"

        if not (os.path.exists(sample["cam0"]) and os.path.exists(sample["cam1"])):
            return None, "no_cam_images"

        cam0_b = load_image_bytes_png(sample["cam0"])
        cam1_b = load_image_bytes_png(sample["cam1"])
        bridge_b = make_pm_bridge_bytes(a0, a1)
        if not (cam0_b and cam1_b and bridge_b):
            return None, "image_load_error"

        instruction = VISUAL_ONLY_THINK_SYSTEM_PROMPT + "\n" + sample["question"]
        # Convert "<answer>X</answer>" gpt response into the visual-only output split.
        gpt = sample["gpt"].strip()
        return {
            "image_list":       [cam0_b, cam1_b, bridge_b],
            "instruction_list": [instruction],
            "output_text_list": [
                "<image_start>",
                f"<image_end>{gpt}",
            ],
            "sample_id": sample["sample_id"],
        }, "ok"
    except Exception as e:
        print(f"[ERROR] {sample.get('sample_id', '?')}: {e}")
        return None, "exception"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference_jsonl", default=DEFAULT_REFERENCE_JSONL)
    p.add_argument("--pm_v4_root",    default=PM_ANNO_ROOTS["v4"])
    p.add_argument("--pm_ankur_root", default=PM_ANNO_ROOTS["ankur"])
    p.add_argument("--pm_v5_root",    default=PM_ANNO_ROOTS["v5"])
    p.add_argument("--image_prefix",  default=DEFAULT_IMAGE_PREFIX,
                   help="Prefix for relative image paths in no_thinking.jsonl")
    p.add_argument("--output_dir",    default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max_workers",   type=int, default=16)
    p.add_argument("--rows_per_group", type=int, default=100)
    p.add_argument("--groups_per_file", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0,
                   help="Process at most N samples (0 = no limit)")
    args = p.parse_args()

    random.seed(args.seed)
    pm_anno_roots = {
        "v4":    args.pm_v4_root,
        "ankur": args.pm_ankur_root,
        "v5":    args.pm_v5_root,
    }

    print(f"Reference jsonl: {args.reference_jsonl}")
    print(f"PM anno roots:")
    for k, v in pm_anno_roots.items():
        print(f"  {k}: {v}")

    samples = []
    with open(args.reference_jsonl) as f:
        for ln in f:
            s = parse_no_thinking_sample(ln, args.image_prefix)
            if s is not None:
                samples.append(s)
    print(f"\nLoaded {len(samples)} non-rotation samples from reference jsonl")

    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
        print(f"  --limit set: keeping first {len(samples)}")

    src_counts = Counter(s["source"] for s in samples)
    print(f"By source: {dict(src_counts)}")

    print(f"\nProcessing {len(samples)} samples with {args.max_workers} workers...")
    process_fn = partial(process_sample, pm_anno_roots=pm_anno_roots)
    status_counts = Counter()
    rows = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for result, status in tqdm(ex.map(process_fn, samples), total=len(samples)):
            status_counts[status] += 1
            if result is not None:
                rows.append(result)
    print(f"Status: {dict(status_counts)}")
    print(f"Successful rows: {len(rows)}/{len(samples)}")

    if not rows:
        print("No samples to write. Exiting.")
        return

    random.shuffle(rows)
    for r in rows:
        r.pop("sample_id", None)

    os.makedirs(args.output_dir, exist_ok=True)
    schema = pa.schema([
        pa.field("image_list",       pa.list_(pa.binary())),
        pa.field("instruction_list", pa.list_(pa.string())),
        pa.field("output_text_list", pa.list_(pa.string())),
    ])

    rows_per_file = args.rows_per_group * args.groups_per_file
    file_index = 0
    parquet_info = {}

    print(f"\nWriting parquet files to {args.output_dir}")
    for i in tqdm(range(0, len(rows), rows_per_file)):
        chunk = rows[i:i + rows_per_file]
        path = os.path.join(args.output_dir, f"chunk_{file_index}.parquet")
        file_index += 1
        n_groups = 0
        with pq.ParquetWriter(path, schema=schema, version="2.6") as writer:
            for j in range(0, len(chunk), args.rows_per_group):
                grp = chunk[j:j + args.rows_per_group]
                writer.write_table(pa.Table.from_pylist(grp, schema=schema))
                n_groups += 1
        parquet_info[path] = {"num_row_groups": n_groups, "num_rows": len(chunk)}

    info_path = os.path.join(args.output_dir, "parquet_info.json")
    with open(info_path, "w") as f:
        json.dump(parquet_info, f, indent=2)

    print(f"\nDone! {len(rows)} samples → {file_index} parquet file(s) in {args.output_dir}")
    print(f"Parquet info: {info_path}")


if __name__ == "__main__":
    main()
