#!/usr/bin/env python3
"""
Build three paired SFT files from a flat ``scene_index.json`` (produced by
``build_map_scene_index.py``). The same set of sample_ids appears in all
three modes:

  no_think                — JSONL (ShareGPT). 3 input images (cam0, cam1,
                            target map). Model answers Yes/No directly.
  visual_think            — Parquet. 3 input images + panorama bridge between
                            <image_start> </image_end>.
  visual_think_topdown    — Parquet. 3 input images + correct top-down bridge
                            between <image_start> </image_end>.

A scene contributes BOTH a positive (target = correct top-down, "Yes") and a
negative (target = a uniformly-sampled perturbed top-down for the same
``asking_to``, "No") row, so the dataset is perfectly 50/50 by construction.

A scene is kept iff (a) cam0/cam1 exist, (b) the per-scene
``correct_agent_{N}.png`` exists for the asker, (c) at least one perturbed map
exists in the scene's ``index.json``, and (d) a panorama is found under one of
the three panorama roots (rendered_panorama_train | rendered_panorama_v4 |
rendered_panorama_v5).

Output layout (under --output_dir):
    no_think/no_think.jsonl + manifest.json
    visual_think/chunk_*.parquet + parquet_info.json + manifest.json
    visual_think_topdown/chunk_*.parquet + parquet_info.json + manifest.json
    pairing.json                     — master sample_id list (in row order)

Usage:
    python create_map_questions_sft_data.py \
        --scene_index /path/to/scratch/VisualCoT/training_data/map_questions_synth/scene_index.json \
        --output_dir /path/to/scratch/VisualCoT/training_data/map_questions_sft
"""

import argparse
import glob
import io
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


NO_THINKING_SYSTEM_PROMPT = (
    "Answer the question directly. "
    "Provide your final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)

VISUAL_ONLY_THINK_SYSTEM_PROMPT = (
    "Think visually to answer the question. "
    "Enclose your visual thinking within <image_start> </image_end>."
)

PANORAMA_ROOTS = [
    "/path/to/scratch/VisualCoT/infinigen/rendered_panorama_train",
    "/path/to/scratch/infinigen/rendered_panorama_v4",
    "/path/to/scratch/infinigen/rendered_panorama_v5",
]
PERTURBED_ROOT = "/path/to/scratch/infinigen/map_questions_perturbed_v3"

DEFAULT_SCENE_INDEX = (
    "/path/to/scratch/VisualCoT/training_data/map_questions_synth/scene_index.json"
)
DEFAULT_OUTPUT_DIR = "/path/to/scratch/VisualCoT/training_data/map_questions_sft"

# Held-out eval benchmarks (5 files × 250 samples each). Any scene_id appearing
# in any of these MUST be excluded from training. Already enforced at render
# time, re-checked here as defense in depth.
DEFAULT_EXCLUDE_FILES = [
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_dataset_map_questions.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance.json",
]


def load_excluded_scene_ids(exclude_files):
    excluded = set()
    for f in exclude_files:
        if not os.path.exists(f):
            print(f"[WARN] exclude file not found, skipping: {f}")
            continue
        d = json.load(open(f))
        for s in d:
            if "scene_id" in s:
                excluded.add(s["scene_id"])
    return excluded


def remap_path(path):
    if path is None:
        return None
    path = path.replace("/path/to/scratch", "/path/to/scratch")
    path = path.replace("/path/to/scratch", "/path/to/scratch")
    return path


def image_to_bytes(image_path):
    if image_path.lower().endswith(".png"):
        with open(image_path, "rb") as f:
            return f.read()
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def find_panorama(scene_id):
    for root in PANORAMA_ROOTS:
        d = os.path.join(root, scene_id)
        if not os.path.isdir(d):
            continue
        hits = glob.glob(os.path.join(d, "panorama_blender_limits_*.png"))
        if hits:
            return sorted(hits)[0]
    return None


def correct_topdown_path(scene_id, asking_to):
    n = 1 if asking_to == "agent_1" else 2
    return os.path.join(PERTURBED_ROOT, scene_id, f"correct_agent_{n}.png")


def build_full_question(row):
    q = row.get("question_both_views") or row.get("question") or ""
    options = row.get("options_user_2") or ["Yes", "No"]
    opts_str = "\n".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(options))
    return f"{q.strip()}\n\n{opts_str}"


def _build_one(row, target_map_path, label, image_root_dir, panorama_path,
               correct_top_path, cached_bytes):
    """Build (no_think, visual_think, visual_think_topdown, sample_id) for a
    given (row, target_map, label) triplet. ``cached_bytes`` is a dict the
    caller fills in to avoid re-reading cam0/cam1/correct_top/panorama bytes
    when emitting both pos and neg rows for the same scene."""
    scene_id = row["scene_id"]
    base_sid = row.get("source_sample_id") or row.get("sample_id")
    if base_sid and (base_sid.endswith("_pos") or base_sid.endswith("_neg")):
        base_sid = base_sid[:-4]
    sid = f"map_synth_{scene_id}_{base_sid}_{label}" if base_sid else f"map_synth_{scene_id}_{label}"
    correct_idx = 0 if label == "pos" else 1
    ans_letter = chr(65 + correct_idx)

    cam0 = remap_path(row["user_1_image_local_path"])
    cam1 = remap_path(row["user_2_image_local_path"])

    full_q = build_full_question(row)

    rel_cam0 = os.path.relpath(cam0, image_root_dir)
    rel_cam1 = os.path.relpath(cam1, image_root_dir)
    rel_target = os.path.relpath(target_map_path, image_root_dir)
    user_msg = f"{NO_THINKING_SYSTEM_PROMPT}\n\n<image><image><image>\n{full_q}"
    no_think = {
        "conversations": [
            {"from": "human", "value": user_msg},
            {"from": "gpt", "value": f"<answer>{ans_letter}</answer>"},
        ],
        "image": [rel_cam0, rel_cam1, rel_target],
        "id": sid,
    }

    if "cam0" not in cached_bytes:
        cached_bytes["cam0"] = image_to_bytes(cam0)
        cached_bytes["cam1"] = image_to_bytes(cam1)
        cached_bytes["correct_top"] = image_to_bytes(correct_top_path)
        cached_bytes["panorama"] = image_to_bytes(panorama_path)
    cam0_b = cached_bytes["cam0"]
    cam1_b = cached_bytes["cam1"]
    correct_top_b = cached_bytes["correct_top"]
    panorama_b = cached_bytes["panorama"]

    if label == "pos":
        target_b = correct_top_b
    else:
        target_b = image_to_bytes(target_map_path)

    if not all([cam0_b, cam1_b, target_b, correct_top_b, panorama_b]):
        return None

    instruction = VISUAL_ONLY_THINK_SYSTEM_PROMPT + "\n" + full_q
    output_text_list = ["<image_start>", f"<image_end><answer>{ans_letter}</answer>"]
    visual_think = {
        "image_list": [cam0_b, cam1_b, target_b, panorama_b],
        "instruction_list": [instruction],
        "output_text_list": output_text_list,
        "sample_id": sid,
    }
    visual_think_topdown = {
        "image_list": [cam0_b, cam1_b, target_b, correct_top_b],
        "instruction_list": [instruction],
        "output_text_list": output_text_list,
        "sample_id": sid,
    }
    return no_think, visual_think, visual_think_topdown, sid


def build_triples_for_scene(row, image_root_dir, perturbed_root, seed_offset=0):
    """For a single synth row, emit BOTH a pos and a neg triple from the
    scene's index.json. Returns a list of up to two (no_think, vt, vtt, sid)
    triples, or [] if any required asset is missing."""
    scene_id = row["scene_id"]
    asking_to = row.get("asking_to", "agent_2")
    cam0 = remap_path(row["user_1_image_local_path"])
    cam1 = remap_path(row["user_2_image_local_path"])
    correct_top = correct_topdown_path(scene_id, asking_to)
    panorama = find_panorama(scene_id)

    if not all(p and os.path.exists(p) for p in [cam0, cam1, correct_top]):
        return []
    if panorama is None or not os.path.exists(panorama):
        return []

    idx_path = os.path.join(perturbed_root, scene_id, "index.json")
    if not os.path.exists(idx_path):
        return []
    try:
        idx = json.load(open(idx_path))
    except Exception:
        return []
    ag = idx.get("agents", {}).get(asking_to)
    if not ag or not ag.get("perturbed_maps"):
        return []

    pert_maps = [pm for pm in ag["perturbed_maps"] if os.path.exists(pm["path"])]
    if not pert_maps:
        return []

    rng = random.Random(hash((scene_id, asking_to, seed_offset)) & 0xFFFFFFFF)
    chosen_neg = rng.choice(pert_maps)["path"]

    cached = {}
    out = []
    pos = _build_one(row, correct_top, "pos", image_root_dir, panorama, correct_top, cached)
    if pos is not None:
        out.append(pos)
    neg = _build_one(row, chosen_neg, "neg", image_root_dir, panorama, correct_top, cached)
    if neg is not None:
        out.append(neg)
    return out


def write_parquet_chunks(rows, out_dir, rows_per_group, groups_per_file):
    schema = pa.schema([
        pa.field("image_list", pa.list_(pa.binary())),
        pa.field("instruction_list", pa.list_(pa.string())),
        pa.field("output_text_list", pa.list_(pa.string())),
    ])
    rows_per_file = rows_per_group * groups_per_file
    parquet_info = {}
    file_index = 0
    # Clean up stale chunk files from prior runs so on-disk state matches manifest.
    for stale in glob.glob(os.path.join(out_dir, "chunk_*.parquet")):
        os.remove(stale)
    payload = [{k: r[k] for k in ("image_list", "instruction_list", "output_text_list")} for r in rows]
    for i in range(0, len(payload), rows_per_file):
        chunk = payload[i:i + rows_per_file]
        path = os.path.join(out_dir, f"chunk_{file_index}.parquet")
        file_index += 1
        ngroups = 0
        with pq.ParquetWriter(path, schema=schema, version="2.6") as writer:
            for j in range(0, len(chunk), rows_per_group):
                group = chunk[j:j + rows_per_group]
                writer.write_table(pa.Table.from_pylist(group, schema=schema))
                ngroups += 1
        parquet_info[path] = {"num_row_groups": ngroups, "num_rows": len(chunk)}
    with open(os.path.join(out_dir, "parquet_info.json"), "w") as f:
        json.dump(parquet_info, f, indent=2)
    return parquet_info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_index", default=DEFAULT_SCENE_INDEX,
                   help="Flat scene_index.json from build_map_scene_index.py.")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--image_root_dir", default="/network/scratch",
                   help="Root for relative image paths in no_think.jsonl.")
    p.add_argument("--exclude_files", nargs="*", default=DEFAULT_EXCLUDE_FILES,
                   help="JSON files whose scene_ids must be excluded (held-out eval benchmarks).")
    p.add_argument("--rows_per_group", type=int, default=10)
    p.add_argument("--groups_per_file", type=int, default=10)
    p.add_argument("--max_workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    print(f"Loading {args.scene_index}")
    with open(args.scene_index) as f:
        scene_index = json.load(f)
    scenes = scene_index["scenes"] if isinstance(scene_index, dict) else scene_index
    print(f"  {len(scenes)} scenes in index")

    # The builder already de-dupes by scene; keep a defensive sort for stable order.
    synth_unique = sorted(scenes, key=lambda r: r["scene_id"])

    # Step 1a: eval-leakage filter (defense in depth)
    excluded = load_excluded_scene_ids(args.exclude_files)
    print(f"  Excluded {len(excluded)} held-out scene_ids from eval benchmarks.")
    before_eval = len(synth_unique)
    synth_unique = [r for r in synth_unique if r["scene_id"] not in excluded]
    print(f"  After eval-leakage filter: {len(synth_unique)} scenes (dropped {before_eval - len(synth_unique)})")

    # Step 1b: panorama-availability filter (drop scenes with no panorama)
    before_pano = len(synth_unique)
    panorama_kept = []
    for r in synth_unique:
        if find_panorama(r["scene_id"]) is not None:
            panorama_kept.append(r)
    synth_unique = panorama_kept
    print(f"  After panorama-availability filter: {len(synth_unique)} scenes "
          f"(dropped {before_pano - len(synth_unique)})")

    # Step 2: upsample pos+neg per kept scene
    print(f"Building pos+neg 3-mode triples (workers={args.max_workers}) ...")

    def _per_scene(r):
        return build_triples_for_scene(r, args.image_root_dir, PERTURBED_ROOT)

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        results = list(tqdm(ex.map(_per_scene, synth_unique), total=len(synth_unique)))

    triples = [t for batch in results for t in batch]
    n_scenes_kept = sum(1 for batch in results if batch)
    n_scenes_drop = len(synth_unique) - n_scenes_kept
    n_kept = len(triples)
    print(f"  kept {n_scenes_kept} scenes ({n_scenes_drop} dropped), {n_kept} total triples")

    if n_kept == 0:
        print("Nothing to write; aborting.")
        return

    rng = random.Random(args.seed)
    rng.shuffle(triples)

    no_thinks = [t[0] for t in triples]
    visual_thinks = [t[1] for t in triples]
    visual_topdowns = [t[2] for t in triples]
    sample_ids = [t[3] for t in triples]

    pos = sum(1 for r in no_thinks if r["conversations"][1]["value"] == "<answer>A</answer>")
    neg = n_kept - pos
    print(f"  pos (Yes / A): {pos}    neg (No / B): {neg}")

    nt_dir = os.path.join(args.output_dir, "no_think")
    vt_dir = os.path.join(args.output_dir, "visual_think")
    vtt_dir = os.path.join(args.output_dir, "visual_think_topdown")
    os.makedirs(nt_dir, exist_ok=True)
    os.makedirs(vt_dir, exist_ok=True)
    os.makedirs(vtt_dir, exist_ok=True)

    nt_path = os.path.join(nt_dir, "no_think.jsonl")
    print(f"Writing {nt_path}")
    with open(nt_path, "w") as f:
        for r in no_thinks:
            f.write(json.dumps(r) + "\n")

    print(f"Writing parquet chunks → {vt_dir}")
    write_parquet_chunks(visual_thinks, vt_dir, args.rows_per_group, args.groups_per_file)

    print(f"Writing parquet chunks → {vtt_dir}")
    write_parquet_chunks(visual_topdowns, vtt_dir, args.rows_per_group, args.groups_per_file)

    manifest = {"total": n_kept, "pos": pos, "neg": neg, "source_sample_ids": sample_ids}
    for d in (nt_dir, vt_dir, vtt_dir):
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    pairing = {"sample_ids_in_order": sample_ids, "total": n_kept}
    with open(os.path.join(args.output_dir, "pairing.json"), "w") as f:
        json.dump(pairing, f, indent=2)

    print(f"\nDone.\n  no_think         → {nt_path}")
    print(f"  visual_think     → {vt_dir}")
    print(f"  visual_think_topdown → {vtt_dir}")
    print(f"  pairing.json     → {os.path.join(args.output_dir, 'pairing.json')}")


if __name__ == "__main__":
    main()
