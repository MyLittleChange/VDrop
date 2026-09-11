#!/usr/bin/env python3
"""
T_cor v2 — annotate co-visible objects with point-matching style colored
dots on V1 + V2 for the COSMIC test split, using the project's canonical
shared-object renderer (annotate_shared_objects.py).

Output: <output_root>/<scene_id>/{annotated_cam0,annotated_cam1}.png
        + annotation.json    (per-scene, not per-sample)

Reuses process_scene from
SpatialUnderstanding/data_creation/rendering/annotate_shared_objects.py.
Skips scenes already annotated (resumable).
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

RENDER_DIR = "/path/to/ThinkMorph-BAGEL-release/SpatialUnderstanding/data_creation/rendering"
sys.path.insert(0, RENDER_DIR)

from annotate_shared_objects import process_scene  # noqa: E402

TEST_JSONS = [
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
]


def find_scene_dir(sample):
    img = Path(sample["user_1_image_local_path"])
    for p in img.parents:
        if (p / "visible_objects.json").exists():
            return p
    return None


def collect_unique_scenes():
    seen = {}
    for jp in TEST_JSONS:
        for x in json.load(open(jp)):
            sid = x["scene_id"]
            if sid in seen:
                continue
            scene_dir = find_scene_dir(x)
            if scene_dir is None:
                continue
            seen[sid] = scene_dir
    return seen  # {scene_id: scene_dir}


def _worker(args):
    scene_dir, output_root = args
    try:
        return process_scene(
            scene_dir=Path(scene_dir),
            output_root=Path(output_root),
            source="ankur",
            data_pass="auto",
            marker_radius=16,
        )
    except Exception as e:
        return {"scene_id": Path(scene_dir).name, "status": "exception", "reason": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output_root",
        default="/path/to/scratch/VisualCoT/infinigen/correspondence_shared_objects_approved_mcqs",
    )
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=None)
    args = ap.parse_args()

    scenes_map = collect_unique_scenes()
    items = sorted(scenes_map.items())
    if args.end_idx is None:
        args.end_idx = len(items)
    items = items[args.start_idx : args.end_idx]
    print(f"unique scenes total: {len(scenes_map)}, processing {len(items)} (idx {args.start_idx}..{args.end_idx})")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Skip already-done scenes
    todo = []
    n_skipped = 0
    for sid, sdir in items:
        if (out_root / sid / "annotation.json").exists():
            n_skipped += 1
            continue
        todo.append((str(sdir), str(out_root)))
    print(f"already done: {n_skipped}, to process: {len(todo)}")

    counters = {"ok": 0, "failed": 0, "exception": 0, "skipped": n_skipped}
    if not todo:
        print("nothing to do.")
        return

    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futs = [ex.submit(_worker, t) for t in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            st = r.get("status", "unknown")
            counters[st] = counters.get(st, 0) + 1
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}: {counters}")

    print(f"final: {counters}")


if __name__ == "__main__":
    main()
