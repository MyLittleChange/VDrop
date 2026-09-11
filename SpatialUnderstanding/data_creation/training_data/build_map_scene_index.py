#!/usr/bin/env python3
"""
Build a flat scene_index.json for the map-question SFT pipeline by
filesystem-walking the three Infinigen scene roots:

  V4:    /path/to/scratch/infinigen/outputs_rendered/<room_part>/<scene>/
  V5:    /path/to/scratch/infinigen/spatial/<room_part>/<scene>/
  Ankur: /path/to/scratch/infinigen/infinigen_debang/infinigen/v00_filtered/<vN>/<room_part>/<scene>/  (read-only)

A scene is kept iff:
  - it has a cam-pair on disk at frames/Image/camera_0/Image_{0,1}_*_*.png
  - <PERTURBED_ROOT>/<scene_id>/index.json exists, is complete, and has at
    least one agent slot with both correct_map_path and >=1 perturbed_maps
  - scene_id is not in the held-out eval-leakage union (5 approved JSONs)

For each kept scene we pick ``asking_to`` deterministically (the agent slot
with the larger perturbed_maps count; tie-break ``agent_2``). The asker's cam
is ``user_2_image_local_path`` (Ankur convention).

Output:
    {
        "scenes": [
            {
              "scene_id": "...",
              "room_part": "...",
              "asking_to": "agent_2" | "agent_1",
              "user_1_image_local_path": "...",  # non-asker
              "user_2_image_local_path": "...",  # asker
              "source_root": "v4" | "v5" | "ankur",
              "question_both_views": "From the perspective of the second image, ..."
            },
            ...
        ],
        "n_scenes": <int>
    }

Usage:
    python build_map_scene_index.py \\
        --output /path/to/scratch/VisualCoT/training_data/map_questions_synth/scene_index.json

    # Drop V4 (non-rectangular floor plans render incorrectly):
    python build_map_scene_index.py \\
        --exclude_sources v4 \\
        --output /path/to/scratch/VisualCoT/training_data/map_questions_synth/scene_index.json
"""

import argparse
import glob
import json
import os
from collections import Counter


V4_RENDER_ROOT = "/path/to/scratch/infinigen/outputs_rendered"
V5_RENDER_ROOT = "/path/to/scratch/infinigen/spatial"
ANKUR_RENDER_ROOT = "/path/to/scratch/infinigen/infinigen_debang/infinigen/v00_filtered"
DEFAULT_PERTURBED_ROOT = "/path/to/scratch/infinigen/map_questions_perturbed"

DEFAULT_EXCLUDE_FILES = [
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_dataset_map_questions.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance.json",
]

QUESTION_BOTH_VIEWS = (
    "From the perspective of the second image, is this top-down map of the room correct?"
)


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
    print(f"Excluded {len(excluded)} held-out scene IDs.")
    return excluded


def load_perturbed_index(perturbed_root, scene_id):
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


def find_cam_pair(scene_dir):
    """Return (cam0_path, cam1_path) or (None, None) if either image is missing."""
    cam_dir = os.path.join(scene_dir, "frames", "Image", "camera_0")
    if not os.path.isdir(cam_dir):
        return None, None
    cam0 = sorted(glob.glob(os.path.join(cam_dir, "Image_0_*.png")))
    cam1 = sorted(glob.glob(os.path.join(cam_dir, "Image_1_*.png")))
    if not cam0 or not cam1:
        return None, None
    return cam0[0], cam1[0]


def discover_scenes_under(root, depth):
    """Yield (scene_id, room_part, scene_dir, source_tag) for each scene directory.

    depth=2: <root>/<room_part>/<scene>/   (V4, V5)
    depth=3: <root>/<vN>/<room_part>/<scene>/   (Ankur)
    """
    if not os.path.isdir(root):
        return
    if depth == 2:
        for room_part in sorted(os.listdir(root)):
            rp_dir = os.path.join(root, room_part)
            if not os.path.isdir(rp_dir):
                continue
            for scene_id in sorted(os.listdir(rp_dir)):
                scene_dir = os.path.join(rp_dir, scene_id)
                if os.path.isdir(scene_dir):
                    yield scene_id, room_part, scene_dir
    elif depth == 3:
        for vN in sorted(os.listdir(root)):
            vN_dir = os.path.join(root, vN)
            if not os.path.isdir(vN_dir):
                continue
            for room_part in sorted(os.listdir(vN_dir)):
                rp_dir = os.path.join(vN_dir, room_part)
                if not os.path.isdir(rp_dir):
                    continue
                for scene_id in sorted(os.listdir(rp_dir)):
                    scene_dir = os.path.join(rp_dir, scene_id)
                    if os.path.isdir(scene_dir):
                        yield scene_id, f"{vN}/{room_part}", scene_dir


def pick_asking_to(idx):
    """Deterministically pick the agent slot to ask about. Prefer the slot
    with more perturbed maps (= more negative-training signal). Tie-break
    agent_2 (matches Ankur's normalized-schema convention)."""
    agents = idx.get("agents", {})
    candidates = []
    for slot in ("agent_1", "agent_2"):
        ag = agents.get(slot, {})
        if ag.get("correct_map_path") and ag.get("perturbed_maps"):
            candidates.append((slot, len(ag["perturbed_maps"])))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[1], 0 if x[0] == "agent_2" else 1))
    return candidates[0][0]


def build_scene_index(perturbed_root, excluded_scene_ids, exclude_sources=None):
    exclude_sources = set(exclude_sources or [])
    sources = [
        ("v4", V4_RENDER_ROOT, 2),
        ("v5", V5_RENDER_ROOT, 2),
        ("ankur", ANKUR_RENDER_ROOT, 3),
    ]
    scenes = []
    seen = set()
    stats = Counter()
    drop = Counter()

    for tag, root, depth in sources:
        if tag in exclude_sources:
            print(f"Skipping {tag} (excluded by --exclude_sources)")
            continue
        print(f"Scanning {tag}: {root}")
        n_seen = n_kept = 0
        for scene_id, room_part, scene_dir in discover_scenes_under(root, depth):
            n_seen += 1
            if scene_id in seen:
                drop[f"{tag}:dup_scene"] += 1
                continue
            if scene_id in excluded_scene_ids:
                drop[f"{tag}:eval_leak"] += 1
                continue

            idx = load_perturbed_index(perturbed_root, scene_id)
            if idx is None:
                drop[f"{tag}:no_index"] += 1
                continue

            asking_to = pick_asking_to(idx)
            if asking_to is None:
                drop[f"{tag}:no_perturbed"] += 1
                continue

            cam0, cam1 = find_cam_pair(scene_dir)
            if not cam0 or not cam1:
                drop[f"{tag}:no_cam_pair"] += 1
                continue

            # Asker convention: asker's cam is user_2_*. cam0 is "agent 1's
            # view", cam1 is "agent 2's view" (the renderer convention is
            # Image_<n>_<...>.png where n is the camera index).
            if asking_to == "agent_1":
                u1, u2 = cam1, cam0
            else:
                u1, u2 = cam0, cam1

            scenes.append({
                "scene_id": scene_id,
                "room_part": room_part,
                "asking_to": asking_to,
                "user_1_image_local_path": u1,
                "user_2_image_local_path": u2,
                "source_root": tag,
                "question_both_views": QUESTION_BOTH_VIEWS,
            })
            seen.add(scene_id)
            n_kept += 1
            stats[tag] += 1
        print(f"  scanned={n_seen} kept={n_kept}")

    print("\nDrop reasons:")
    for k, v in sorted(drop.items()):
        print(f"  {k:30s}  {v}")

    print("\nKept by source:")
    for k, v in sorted(stats.items()):
        print(f"  {k:6s}  {v}")
    print(f"  TOTAL   {len(scenes)}")
    return scenes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--perturbed_root", default=DEFAULT_PERTURBED_ROOT)
    p.add_argument("--exclude_files", nargs="*", default=DEFAULT_EXCLUDE_FILES)
    p.add_argument(
        "--exclude_sources",
        nargs="*",
        default=[],
        choices=["v4", "v5", "ankur"],
        help="Source tags to skip entirely (e.g. 'v4' to drop V4 scenes whose "
             "non-rectangular floor plans render incorrectly).",
    )
    p.add_argument("--output", required=True)
    args = p.parse_args()

    excluded = load_excluded_scene_ids(args.exclude_files)
    scenes = build_scene_index(args.perturbed_root, excluded, args.exclude_sources)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"scenes": scenes, "n_scenes": len(scenes)}, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
