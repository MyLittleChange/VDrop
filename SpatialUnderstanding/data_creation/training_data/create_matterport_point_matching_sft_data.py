#!/usr/bin/env python3
"""
Create point-matching SFT training data from matterport_views.

Directory layout:
    <data_root>/<scene_id>/<viewpoint_id>/
        rotation_qa_questions_point_match.json
        point_match/
            pointmatch_NNN_image1.png
            pointmatch_NNN_image2.png
            pointmatch_NNN_panorama.png   ← visual-thinking bridge

Four thinking modes:

  --thinking_mode visual_only (default)
    [image1, image2, panorama] → panorama is the visual-thinking target.
    Output: sharded parquet files.

  --thinking_mode no_thinking
    [image1, image2] → direct answer.
    Output: JSONL in ShareGPT format.

  --thinking_mode json_export
    Copy images to images/ and emit a flat JSON for manual inspection.

  --thinking_mode text_thinking
    Reads pre-annotated matterport_point_matching.jsonl from
    --text_thinking_source_file and writes text_thinking.jsonl.

  --thinking_mode interleaved
    Combines text-think + visual-think + answer. Same images as visual_only;
    prepends <think>{text}</think> to the first output text. Skips samples
    missing a think trace. Output: sharded parquet in <output_dir>/interleaved/.

Usage:
    python create_matterport_point_matching_sft_data.py
    python create_matterport_point_matching_sft_data.py --thinking_mode no_thinking
    python create_matterport_point_matching_sft_data.py --n_samples 500
"""

import argparse
import io
import json
import os
import random
import re
import shutil
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

NO_THINKING_SYSTEM_PROMPT = (
    "Answer the question directly. "
    "Provide your final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)

TEXT_THINKING_SYSTEM_PROMPT = (
    "Think step by step before answering. "
    "Enclose your thinking within <think> </think> tags. "
    "Provide your final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)

INTERLEAVED_SYSTEM_PROMPT = (
    "Think step by step before answering, and use a visual sketch to support your reasoning. "
    "Enclose your text thinking within <think> </think> tags, then your visual thinking within "
    "<image_start> <image_end>. "
    "Provide your final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _extract_human_question(human_val: str) -> str:
    last = human_val.rfind("<image>")
    tail = human_val[last + len("<image>"):] if last != -1 else human_val
    return tail.lstrip("\n").strip()


def build_think_lookup(jsonl_path: str):
    """Returns {(sample_id, question_text): think_text}."""
    lookup = {}
    if not os.path.exists(jsonl_path):
        print(f"  [warn] missing think source {jsonl_path}")
        return lookup
    with open(jsonl_path) as f:
        for ln in f:
            row = json.loads(ln)
            sid = row.get("id")
            convs = row.get("conversations", [])
            human = next((c["value"] for c in convs if c.get("from") == "human"), "")
            gpt   = next((c["value"] for c in convs if c.get("from") == "gpt"),   "")
            m = THINK_RE.search(gpt)
            if not (sid and m):
                continue
            lookup[(sid, _extract_human_question(human))] = m.group(1).strip()
    return lookup


def lookup_think(think_lookup, sample_id, question):
    qn = (question or "").strip()
    sid = sample_id or ""
    if (sid, qn) in think_lookup:
        return think_lookup[(sid, qn)]
    for (k_sid, k_q), think in think_lookup.items():
        if k_sid != sid:
            continue
        if k_q.startswith(qn) or qn.startswith(k_q):
            return think
    return None

DEFAULT_DATA_ROOT = "/path/to/scratch/matterport_views"
DEFAULT_OUTPUT_DIR = "/path/to/scratch/matterport/training_data_point_matching_matterport"
QA_FILENAME = "rotation_qa_questions_point_match.json"


# ── Image loading ─────────────────────────────────────────────────────────────

def image_to_bytes(image_path: str) -> bytes:
    """Load image bytes. For PNGs, read raw (no re-encode)."""
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


def answer_for_sample(sample: dict) -> str:
    ci = sample.get("correct_index")
    if isinstance(ci, int):
        return chr(65 + ci)
    return sample.get("correct_answer", "")


# ── Loading / sampling ────────────────────────────────────────────────────────

def load_samples(data_root: str) -> list:
    """Walk <data_root>/<scene_id>/<viewpoint_id>/ and collect every question.

    Drops questions whose image1, image2, or panorama is missing on disk.
    """
    if not os.path.isdir(data_root):
        print(f"[ERROR] data_root not found: {data_root}")
        return []

    samples = []
    n_vp_seen = n_vp_loaded = 0
    for scene_id in sorted(os.listdir(data_root)):
        scene_dir = os.path.join(data_root, scene_id)
        if not os.path.isdir(scene_dir):
            continue
        for viewpoint_id in sorted(os.listdir(scene_dir)):
            vp_dir = os.path.join(scene_dir, viewpoint_id)
            if not os.path.isdir(vp_dir):
                continue
            n_vp_seen += 1
            qa_file = os.path.join(vp_dir, QA_FILENAME)
            if not os.path.exists(qa_file):
                continue
            try:
                with open(qa_file, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  [WARN] Failed to load {qa_file}: {e}")
                continue
            n_vp_loaded += 1
            for i, q in enumerate(data.get("rotation_questions", [])):
                imgs = q.get("images") or {}
                img1_fn = imgs.get("image_1")
                img2_fn = imgs.get("image_2")
                panorama_fn = imgs.get("panorama_view")
                if not (img1_fn and img2_fn and panorama_fn):
                    continue
                # image paths in JSON are relative to vp_dir (already include "point_match/")
                img1_abs = os.path.join(vp_dir, img1_fn)
                img2_abs = os.path.join(vp_dir, img2_fn)
                panorama_abs = os.path.join(vp_dir, panorama_fn)
                if not (os.path.exists(img1_abs) and os.path.exists(img2_abs)
                        and os.path.exists(panorama_abs)):
                    continue
                question_text = q.get("question_text", "")
                options = q.get("options", [])
                # question_text already includes the options block; don't append again
                full_question = question_text
                samples.append({
                    "sample_id":      q.get("question_id",
                                            f"{scene_id}_{viewpoint_id}_pointmatch_{i}"),
                    "scene_id":       scene_id,
                    "viewpoint_id":   viewpoint_id,
                    "question":       full_question,
                    "options":        options,
                    "correct_index":  q.get("correct_index"),
                    "correct_answer": q.get("correct_answer", ""),
                    "question_type":  q.get("question_type", ""),
                    "img1_path":      img1_abs,
                    "img2_path":      img2_abs,
                    "panorama_path":  panorama_abs,
                })

    print(f"Scanned {n_vp_seen} viewpoints, loaded {n_vp_loaded}, "
          f"collected {len(samples)} questions")
    return samples


def sample_pool(pool: list, target: int, tag: str) -> list:
    if target <= 0:
        return list(pool)
    if len(pool) <= target:
        if len(pool) < target:
            print(f"  [WARN] {tag}: requested {target} but only {len(pool)} available; taking all.")
        return list(pool)
    return random.sample(pool, target)


# ── Per-sample processors ─────────────────────────────────────────────────────

def process_item_visual_only(sample: dict):
    """[image1, image2, panorama] with panorama as visual-thinking target."""
    try:
        paths = [sample["img1_path"], sample["img2_path"], sample["panorama_path"]]
        img_bytes_list = []
        for p in paths:
            b = image_to_bytes(p)
            if b is None:
                return None
            img_bytes_list.append(b)
        answer_text = answer_for_sample(sample)
        return {
            "image_list":       img_bytes_list,
            "instruction_list": [VISUAL_ONLY_THINK_SYSTEM_PROMPT + "\n" + sample["question"]],
            "output_text_list": [
                "<image_start>",
                f"<image_end><answer>{answer_text}</answer>",
            ],
            "sample_id": sample["sample_id"],
        }
    except Exception as e:
        print(f"[ERROR] {sample.get('sample_id', '?')}: {e}")
        return None


def process_item_interleaved(sample: dict, think_lookup: dict):
    """[image1, image2, panorama] with text think + visual think + answer."""
    try:
        sample_id = sample["sample_id"]
        question = sample["question"]
        think_text = lookup_think(think_lookup, sample_id, question)
        if think_text is None:
            return None

        paths = [sample["img1_path"], sample["img2_path"], sample["panorama_path"]]
        img_bytes_list = []
        for p in paths:
            b = image_to_bytes(p)
            if b is None:
                return None
            img_bytes_list.append(b)

        answer_text = answer_for_sample(sample)
        return {
            "image_list":       img_bytes_list,
            "instruction_list": [INTERLEAVED_SYSTEM_PROMPT + "\n" + question],
            "output_text_list": [
                f"<think>{think_text}</think><image_start>",
                f"<image_end><answer>{answer_text}</answer>",
            ],
            "sample_id": sample_id,
        }
    except Exception as e:
        print(f"[ERROR] {sample.get('sample_id', '?')}: {e}")
        return None


def process_item_no_thinking(sample: dict, image_root_dir: str):
    """[image1, image2] with direct answer; JSONL ShareGPT format."""
    try:
        img_paths = [sample["img1_path"], sample["img2_path"]]
        for p in img_paths:
            if not os.path.exists(p):
                print(f"[WARN] {sample['sample_id']}: image not found: {p}")
                return None
        rel_paths = [os.path.relpath(p, image_root_dir) for p in img_paths]
        full_question = "<image><image>\n" + sample["question"]
        user_message = f"{NO_THINKING_SYSTEM_PROMPT}\n\n{full_question}"
        answer_text = answer_for_sample(sample)
        return {
            "conversations": [
                {"from": "human", "value": user_message},
                {"from": "gpt",   "value": f"<answer>{answer_text}</answer>"},
            ],
            "image": rel_paths,
            "id":    sample["sample_id"],
        }
    except Exception as e:
        print(f"[ERROR] {sample.get('sample_id', '?')}: {e}")
        return None


def process_item_json_export(sample: dict, images_dir: str):
    """Copy images to images/ and emit a flat record for inspection."""
    try:
        sid = sample["sample_id"]
        img1_dst = os.path.join(images_dir, f"{sid}_img1.png")
        img2_dst = os.path.join(images_dir, f"{sid}_img2.png")
        panorama_dst = os.path.join(images_dir, f"{sid}_panorama.png")
        shutil.copy2(sample["img1_path"], img1_dst)
        shutil.copy2(sample["img2_path"], img2_dst)
        shutil.copy2(sample["panorama_path"], panorama_dst)
        answer_text = answer_for_sample(sample)
        return {
            "sample_id":    sid,
            "scene_id":     sample["scene_id"],
            "viewpoint_id": sample["viewpoint_id"],
            "question_type": sample["question_type"],
            "img1":         f"images/{sid}_img1.png",
            "img2":         f"images/{sid}_img2.png",
            "panorama":     f"images/{sid}_panorama.png",
            "instruction":  VISUAL_ONLY_THINK_SYSTEM_PROMPT + "\n" + sample["question"],
            "output":       f"<image_start><image_end><answer>{answer_text}</answer>",
        }
    except Exception as e:
        print(f"[ERROR] {sample.get('sample_id', '?')}: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Create matterport point-matching SFT data from matterport_views"
    )
    parser.add_argument("--data_root",  default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--thinking_mode",
        choices=["visual_only", "no_thinking", "json_export", "text_thinking", "interleaved"],
        default="visual_only",
    )
    parser.add_argument(
        "--text_thinking_source_file",
        default=None,
        help="Path to pre-annotated matterport_point_matching.jsonl for text_thinking mode. "
             "Defaults to <output_dir>/text_thinking/matterport_point_matching.jsonl",
    )
    parser.add_argument(
        "--interleaved_text_file",
        default=None,
        help="Path to pre-annotated matterport_point_matching.jsonl for interleaved mode. "
             "Defaults to <output_dir>/text_thinking/matterport_point_matching.jsonl",
    )
    parser.add_argument("--n_samples", type=int, default=1500,
                        help="Max samples to use (0 = all)")
    parser.add_argument("--max_workers",     type=int, default=32)
    parser.add_argument("--rows_per_group",  type=int, default=10)
    parser.add_argument("--groups_per_file", type=int, default=10)
    parser.add_argument(
        "--image_root_dir",
        default="/network/scratch",
        help="Root for relative image paths (no_thinking mode only)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample_ids_file",
        default=None,
        help=(
            "Path to a JSON file containing a list of sample_ids to use. "
            "If given, --n_samples and --seed are ignored for selection. "
            "If not given, the selected IDs are saved to "
            "<output_dir>/selected_sample_ids.json so other modes can reuse them."
        ),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = os.path.join(args.output_dir, args.thinking_mode)
    print(f"Thinking mode : {args.thinking_mode}")
    print(f"Data root     : {args.data_root}")
    print(f"Output dir    : {output_dir}")

    # ── text_thinking: load pre-annotated file ───────────────────────────────
    if args.thinking_mode == "text_thinking":
        src = args.text_thinking_source_file or os.path.join(
            args.output_dir, "text_thinking", "matterport_point_matching.jsonl"
        )
        if not os.path.exists(src):
            print(f"[ERROR] source file not found: {src}")
            return
        all_data = [json.loads(ln) for ln in open(src)]
        n_deduped = 0
        for row in all_data:
            convs = row.get("conversations", [])
            if convs and convs[0].get("from") == "human":
                val = convs[0]["value"].replace(NO_THINKING_SYSTEM_PROMPT, TEXT_THINKING_SYSTEM_PROMPT)
                # Strip duplicate trailing options block (question_text already contains options)
                sep = val.rfind("\n\n")
                if sep != -1:
                    trailing = val[sep + 2:]
                    before = val[:sep]
                    if before.endswith(trailing):
                        val = before
                        n_deduped += 1
                convs[0]["value"] = val
        if n_deduped:
            print(f"  Removed duplicate options block from {n_deduped} rows")
        random.shuffle(all_data)
        out_dir = os.path.join(args.output_dir, "text_thinking")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "text_thinking.jsonl")
        with open(out_file, "w") as f:
            for item in all_data:
                f.write(json.dumps(item) + "\n")
        print(f"\nDone! {len(all_data)} rows → {out_file}")
        return

    # ── Load ─────────────────────────────────────────────────────────────────
    print("\nLoading point-matching QA...")
    all_samples = load_samples(args.data_root)

    if not all_samples:
        print("No samples found. Exiting.")
        return

    if args.sample_ids_file:
        print(f"Loading sample IDs from {args.sample_ids_file}")
        with open(args.sample_ids_file, "r") as f:
            selected_ids = set(json.load(f))
        all_samples = [s for s in all_samples if s["sample_id"] in selected_ids]
        missing = selected_ids - {s["sample_id"] for s in all_samples}
        if missing:
            print(f"  [WARN] {len(missing)} requested IDs not found in data_root")
        print(f"Using {len(all_samples)} samples (from IDs file)")
    else:
        all_samples = sample_pool(all_samples, args.n_samples, "total")
        print(f"Using {len(all_samples)} samples")
        ids_dir = args.output_dir
        os.makedirs(ids_dir, exist_ok=True)
        ids_file = os.path.join(ids_dir, "selected_sample_ids.json")
        with open(ids_file, "w") as f:
            json.dump([s["sample_id"] for s in all_samples], f, indent=2)
        print(f"Saved selected sample IDs → {ids_file}")

    # Breakdown by scene
    scene_counts = Counter(s["scene_id"] for s in all_samples)
    print(f"Across {len(scene_counts)} scenes")

    # ── Select processor ─────────────────────────────────────────────────────
    if args.thinking_mode == "no_thinking":
        process_fn = partial(process_item_no_thinking, image_root_dir=args.image_root_dir)
    elif args.thinking_mode == "json_export":
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        process_fn = partial(process_item_json_export, images_dir=images_dir)
    elif args.thinking_mode == "interleaved":
        src = args.interleaved_text_file or os.path.join(
            args.output_dir, "text_thinking", "matterport_point_matching.jsonl"
        )
        print(f"\nBuilding think_lookup from {src}")
        think_lookup = build_think_lookup(src)
        print(f"  think_lookup: {len(think_lookup)} (sample_id, question) keys")
        process_fn = partial(process_item_interleaved, think_lookup=think_lookup)
    else:
        process_fn = process_item_visual_only

    print(f"\nProcessing {len(all_samples)} samples with {args.max_workers} workers...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        processed = list(tqdm(executor.map(process_fn, all_samples),
                              total=len(all_samples)))
    all_data = [r for r in processed if r is not None]
    random.shuffle(all_data)
    print(f"Successfully processed: {len(all_data)} samples")

    if not all_data:
        print("No samples to write. Exiting.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── Write output ─────────────────────────────────────────────────────────
    if args.thinking_mode == "no_thinking":
        jsonl_file = os.path.join(output_dir, "no_thinking.jsonl")
        print(f"Writing JSONL to {jsonl_file}")
        with open(jsonl_file, "w") as f:
            for item in all_data:
                f.write(json.dumps(item) + "\n")
        print(f"\nDone! {len(all_data)} samples → {jsonl_file}")

    elif args.thinking_mode == "json_export":
        json_file = os.path.join(output_dir, "training_data.json")
        print(f"Writing JSON to {json_file}")
        with open(json_file, "w") as f:
            json.dump(all_data, f, indent=2)
        print(f"\nDone! {len(all_data)} samples → {json_file}")

    else:  # visual_only → sharded parquet
        for item in all_data:
            item.pop("sample_id", None)

        schema = pa.schema([
            pa.field("image_list",       pa.list_(pa.binary())),
            pa.field("instruction_list", pa.list_(pa.string())),
            pa.field("output_text_list", pa.list_(pa.string())),
        ])

        rows_per_file = args.rows_per_group * args.groups_per_file
        file_index    = 0
        parquet_info  = {}

        print(f"Writing parquet files to {output_dir}")
        for i in tqdm(range(0, len(all_data), rows_per_file)):
            file_data    = all_data[i:i + rows_per_file]
            parquet_file = os.path.join(output_dir, f"chunk_{file_index}.parquet")
            file_index  += 1
            num_row_groups = 0
            with pq.ParquetWriter(parquet_file, schema=schema, version="2.6") as writer:
                for j in range(0, len(file_data), args.rows_per_group):
                    group_data  = file_data[j:j + args.rows_per_group]
                    group_table = pa.Table.from_pylist(group_data, schema=schema)
                    writer.write_table(group_table)
                    num_row_groups += 1
            parquet_info[parquet_file] = {
                "num_row_groups": num_row_groups,
                "num_rows":       len(file_data),
            }

        info_path = os.path.join(output_dir, "parquet_info.json")
        with open(info_path, "w") as f:
            json.dump(parquet_info, f, indent=2)

        print(f"\nDone! {len(all_data)} samples → {file_index} parquet file(s) in {output_dir}")
        print(f"Parquet info: {info_path}")


if __name__ == "__main__":
    main()
