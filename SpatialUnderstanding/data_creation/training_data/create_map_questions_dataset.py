#!/usr/bin/env python3
"""
Synthesize binary "Is this top-down map of the room correct?" map questions from
Infinigen V4 + V5 cam-pair samples plus same-scene PERTURBED top-down maps
pre-rendered by ``render_perturbed_topdowns.py`` (which reuses the perturbation
engine from MultiAgent_Spatial/question_generation_v2/map_godbless_v2.py — the
same engine that produced Ankur's 250-sample evaluation set).

Conventions (verified against approved_dataset_map_questions_normalized.json):
  - ``asking_to`` identifies the asker. In Ankur's normalized map JSON the
    asker's data always sits in ``user_2_*`` (question + image), and the map's
    agent slot matches ``asking_to``.
  - V4/V5 normalized JSONs do NOT follow that convention: they leave the
    asker's data on whichever side ``asking_to`` says (``agent_1`` -> data in
    ``user_1_*``, ``agent_2`` -> data in ``user_2_*``). For files without
    ``asking_to`` (V4 counting / relative_distance / spatial), we infer it from
    which of ``user_1_question`` / ``user_2_question`` is set.
  - To produce Ankur-compatible rows, we therefore SWAP ``user_1`` and
    ``user_2`` image paths whenever the asker is ``agent_1`` so the asker
    always ends up in ``user_2_*``.

For each scene we emit exactly ONE sample: pick one cam-pair row uniformly,
then flip a coin —
  - heads: positive (answer = "Yes") with the scene's own
    ``correct_agent_{N}.png`` where ``N`` matches ``asking_to``.
  - tails: negative (answer = "No") with one of the scene's perturbed maps for
    the same ``asking_to``, sampled uniformly from {type2, type3, counting}
    variants.

A scene must have a complete index entry (correct map + at least one perturbed
map for the row's asking_to) or the row is dropped.

Output: JSON only, in the Ankur schema. A separate script handles parquet.

Usage:
    python create_map_questions_dataset.py \\
        --perturbed_root /path/to/scratch/infinigen/map_questions_perturbed \\
        --output_dir /path/to/scratch/VisualCoT/training_data/map_questions_synth
"""

import argparse
import json
import os
import random
from collections import defaultdict


DEFAULT_SOURCE_JSONS = [
    # V4 (some normalized files are 0-byte; fall back to the non-normalized
    # filtered versions which carry the same fields we need)
    "/path/to/scratch/infinigen/dataset_anchor_questions_filtered_V4_normalized.json",
    "/path/to/scratch/infinigen/outputs_rendered/dataset_counting_questions_filtered_V4.json",
    "/path/to/scratch/infinigen/outputs_rendered/dataset_relative_distance_questions_filtered_V4.json",
    "/path/to/scratch/infinigen/dataset_spatial_questions_filtered_V4_normalized.json",
    "/path/to/scratch/infinigen/dataset_perspective_taking_questions_filtered_V4_normalized.json",
    # V5
    "/path/to/scratch/infinigen/spatial/dataset_anchor_questions_filtered_V5_normalized.json",
    "/path/to/scratch/infinigen/spatial/dataset_counting_questions_filtered_V5_normalized.json",
    "/path/to/scratch/infinigen/spatial/dataset_relative_distance_questions_filtered_V5_normalized.json",
    "/path/to/scratch/infinigen/spatial/dataset_spatial_questions_filtered_V5_normalized.json",
    "/path/to/scratch/infinigen/spatial/dataset_perspective_taking_questions_filtered_V5_normalized.json",
    # Ankur broader pools (provide cam-pair rows for ~1,779 extra scenes that
    # only have metadata in /path/to/scratch/...). The four
    # approved_mcqs_*_normalized.json files are NOT included here because their
    # scene_ids overlap fully with the held-out eval set (filtered via
    # DEFAULT_EXCLUDE_FILES below).
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/anchor_dataset_V_Final_2000.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/counting_dataset_V_Final_2000.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/relative_dataset_V_Final_2000.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/relative_dataset_V_Final_2000_train.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/spatial_dataset_V_Final_2000_normalized.json",
]

DEFAULT_PERTURBED_ROOT = "/path/to/scratch/infinigen/map_questions_perturbed"

# Held-out evaluation benchmarks: union of their scene_ids must never appear
# in the synthesized training data. Covers all 5 benchmark JSONs (map +
# 4 MCQ types: anchor, counting, relative direction, relative distance).
DEFAULT_EXCLUDE_FILES = [
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_dataset_map_questions.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance.json",
]

MAP_QUESTION_PROMPT = "Is this top-down map of the room correct?"
MAP_QUESTION_BOTH_VIEWS = (
    "From the perspective of the second image, is this top-down map of the room correct?"
)


# --- Path helpers ------------------------------------------------------------

def remap_path(path):
    if path is None:
        return None
    path = path.replace("/path/to/scratch", "/path/to/scratch")
    path = path.replace("/path/to/scratch", "/path/to/scratch")
    return path


def _agent_num(asking_to: str) -> int:
    return 1 if asking_to == "agent_1" else 2


def infer_asking_to(row: dict) -> str:
    """Return canonical asking_to for a source row.

    If ``asking_to`` is set, use it. Otherwise fall back to whichever question
    field is set (matches the convention in
    ``text_reasoning_prompts/spatial.py:40-46``)."""
    a = row.get("asking_to")
    if a in ("agent_1", "agent_2"):
        return a
    if row.get("user_1_question"):
        return "agent_1"
    if row.get("user_2_question"):
        return "agent_2"
    return None


def load_perturbed_index(perturbed_root: str, scene_id: str):
    """Read <perturbed_root>/<scene_id>/index.json. Returns parsed dict or None
    if the file is missing or malformed."""
    p = os.path.join(perturbed_root, scene_id, "index.json")
    if not os.path.exists(p):
        return None
    try:
        idx = json.load(open(p))
    except Exception:
        return None
    if not idx.get("complete"):
        return None
    return idx


# --- Building the source pool ------------------------------------------------

def load_excluded_scene_ids(exclude_files: list) -> set:
    excluded = set()
    for f in exclude_files:
        if not os.path.exists(f):
            print(f"[WARN] exclude file not found, skipping: {f}")
            continue
        d = json.load(open(f))
        for s in d:
            if "scene_id" in s:
                excluded.add(s["scene_id"])
    print(f"Excluded {len(excluded)} held-out scene IDs.")
    return excluded


def build_source_pool(source_jsons, perturbed_root, excluded_scene_ids):
    """Walk source JSONs, keep rows that have inferable asking_to, existing
    camera images, AND a complete perturbed-map index for the row's scene with
    >=1 perturbed map for the row's agent slot.

    Returns:
        samples: list of dicts each with sample_id, scene_id, room_part,
                 asking_to, agent_num, user_{1,2}_image_local_path (normalized
                 to "asker in user_2"), correct_map_path, perturbed_maps
                 (list of {"path","kind","idx"}).
    """
    samples = []
    seen_keys = set()  # (sample_id, scene_id, agent_num)
    index_cache = {}  # scene_id -> index dict | None

    for f in source_jsons:
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            print(f"[WARN] empty/missing source: {f}")
            continue
        d = json.load(open(f))
        kept = 0
        for s in d:
            sid = s.get("sample_id")
            scn = s.get("scene_id")
            rp = s.get("room_part")
            if not sid or not scn or not rp:
                continue
            asking_to = infer_asking_to(s)
            if asking_to is None:
                continue
            if scn in excluded_scene_ids:
                continue

            n = _agent_num(asking_to)
            key = (sid, scn, n)
            if key in seen_keys:
                continue

            if scn not in index_cache:
                index_cache[scn] = load_perturbed_index(perturbed_root, scn)
            idx = index_cache[scn]
            if idx is None:
                continue
            ag = idx.get("agents", {}).get(asking_to)
            if not ag or not ag.get("correct_map_path") or not ag.get("perturbed_maps"):
                continue

            src_u1 = remap_path(s.get("user_1_image_local_path"))
            src_u2 = remap_path(s.get("user_2_image_local_path"))
            if not src_u1 or not src_u2:
                continue
            if not (os.path.exists(src_u1) and os.path.exists(src_u2)):
                continue

            # Normalize to Ankur convention: asker's image must end up in
            # user_2_image_local_path. In V4/V5 sources, the asker sits in
            # user_{1 if asking_to==agent_1 else 2}, so swap iff asking_to==agent_1.
            if asking_to == "agent_1":
                out_u1, out_u2 = src_u2, src_u1
            else:
                out_u1, out_u2 = src_u1, src_u2

            samples.append({
                "sample_id": sid,
                "scene_id": scn,
                "room_part": rp,
                "asking_to": asking_to,
                "agent_num": n,
                "user_1_image_local_path": out_u1,
                "user_2_image_local_path": out_u2,
                "correct_map_path": ag["correct_map_path"],
                "perturbed_maps": ag["perturbed_maps"],
            })
            seen_keys.add(key)
            kept += 1
        print(f"  {os.path.basename(f):65s}  kept={kept}")

    return samples


# --- Synthesis ---------------------------------------------------------------

def synthesize_pairs(samples, seed):
    """One sample per scene. For each scene, pick one cam-pair row uniformly,
    then flip a coin to decide pos (Yes / correct map) or neg (No / one of the
    perturbed maps for that asking_to slot). Yields ~50/50 Yes/No."""
    rng = random.Random(seed)
    by_scene = defaultdict(list)
    for s in samples:
        by_scene[s["scene_id"]].append(s)

    out = []
    for scn in sorted(by_scene):
        s = rng.choice(by_scene[scn])
        rp = s["room_part"]
        sid = s["sample_id"]
        asking_to = s["asking_to"]

        is_positive = rng.random() < 0.5
        if is_positive:
            map_path = s["correct_map_path"]
            perturbation_kind = None
            answer_idx, answer_text, label = 0, "Yes", "pos"
        else:
            chosen = rng.choice(s["perturbed_maps"])
            map_path = chosen["path"]
            perturbation_kind = chosen.get("kind")
            answer_idx, answer_text, label = 1, "No", "neg"

        out.append({
            "question_type": "map",
            "sample_id": f"map_synth_{scn}_{sid}_{label}",
            "synth_source_sample_id": sid,
            "scene_id": scn,
            "room_part": rp,
            "user_1_image_local_path": s["user_1_image_local_path"],
            "user_2_image_local_path": s["user_2_image_local_path"],
            "user_1_question": None,
            "user_2_question": MAP_QUESTION_PROMPT,
            "options_user_1": None,
            "options_user_2": ["Yes", "No"],
            "user_1_gt_answer_idx": None,
            "user_1_gt_answer_text": None,
            "user_2_gt_answer_idx": answer_idx,
            "user_2_gt_answer_text": answer_text,
            "correct_index": answer_idx,
            "user_1_goal": "Communicate with your partner to answer the following question correctly.",
            "user_2_goal": "Communicate with your partner to help them answer their question correctly.",
            "question": MAP_QUESTION_PROMPT,
            "question_both_views": MAP_QUESTION_BOTH_VIEWS,
            "asking_to": asking_to,
            "option_categories": ["Yes", "No"],
            "global_map_image": None,
            "user_1_perception": None,
            "user_2_perception": None,
            "panorama_path": None,
            "map_image_path": map_path,
            "topdown_path": map_path,
            "perturbation_kind": perturbation_kind,
        })

    pos = sum(1 for s in out if s["correct_index"] == 0)
    neg = len(out) - pos
    print(f"Synthesized {len(out)} samples ({pos} pos + {neg} neg).")
    return out


# --- Main --------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source_jsons", nargs="*", default=DEFAULT_SOURCE_JSONS)
    p.add_argument("--perturbed_root", default=DEFAULT_PERTURBED_ROOT,
                   help="Root dir written by render_perturbed_topdowns.py (one subdir per scene_id with index.json).")
    p.add_argument("--exclude_files", nargs="*", default=DEFAULT_EXCLUDE_FILES,
                   help="JSON files whose scene_ids are forbidden in the synthesized output.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_samples", type=int, default=2000,
                   help="Cap total synthesized samples. Each scene contributes exactly 1 "
                        "sample (coin-flipped pos/neg), so this caps the scene count too.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)

    excluded = load_excluded_scene_ids(args.exclude_files)
    samples = build_source_pool(args.source_jsons, args.perturbed_root, excluded)
    n_scenes = len({s["scene_id"] for s in samples})
    print(f"\nSource pool: {len(samples)} cam-pair rows across {n_scenes} scenes.")

    synth = synthesize_pairs(samples, args.seed)

    if args.max_samples is not None and len(synth) > args.max_samples:
        rng = random.Random(args.seed)
        rng.shuffle(synth)
        synth = synth[:args.max_samples]
        print(f"Capped to {len(synth)} samples (--max_samples).")

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "map_questions_synth.json")
    with open(json_path, "w") as f:
        json.dump(synth, f, indent=2)
    print(f"Wrote JSON: {json_path}  (n={len(synth)})")

    pos = sum(1 for s in synth if s["correct_index"] == 0)
    neg = sum(1 for s in synth if s["correct_index"] == 1)
    print(f"\nFinal: {len(synth)} samples — {pos} positive (Yes), {neg} negative (No)")
    asking_dist = defaultdict(int)
    kind_dist = defaultdict(int)
    for s in synth:
        asking_dist[s["asking_to"]] += 1
        if s.get("perturbation_kind"):
            kind_dist[s["perturbation_kind"]] += 1
    print(f"asking_to distribution: {dict(asking_dist)}")
    print(f"perturbation_kind distribution (negatives): {dict(kind_dist)}")
    print(f"Held-out leak check: {len(set(s['scene_id'] for s in synth) & excluded)} "
          f"(must be 0)")


if __name__ == "__main__":
    main()
