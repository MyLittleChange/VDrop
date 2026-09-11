#!/usr/bin/env python3
"""
Build T_noise pool + per-question noise assignments for the
Two-Reader Informativeness experiment.

Pool: 100 panoramas drawn from rendered_panorama_train/<scene>, where the
scene_id is NOT in the COSMIC test split's union of scene_ids.

Per-question assignment: deterministic RNG seeded by sample_id -> one pool
index, persisted to noise_assignments.json.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

TEST_JSONS = [
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
]
PANO_TRAIN_ROOT = Path("/path/to/scratch/VisualCoT/infinigen/rendered_panorama_train")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness",
    )
    parser.add_argument("--pool_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Collect test scene_ids
    test_scene_ids = set()
    samples = []
    for jp in TEST_JSONS:
        subtask = Path(jp).stem.replace("approved_mcqs_", "").replace("_normalized", "")
        d = json.load(open(jp))
        for x in d:
            test_scene_ids.add(x["scene_id"])
            samples.append({"sample_id": x["sample_id"], "subtask": subtask})
    print(f"test scene_ids: {len(test_scene_ids)} unique across {len(samples)} samples")

    # 2) Build candidate pool (one panorama per train scene, alphabetically first)
    candidates = []
    for scene_dir in sorted(PANO_TRAIN_ROOT.iterdir()):
        if not scene_dir.is_dir():
            continue
        if scene_dir.name in test_scene_ids:
            continue
        panos = sorted(scene_dir.glob("panorama_blender_limits_*.png"))
        if panos:
            candidates.append(str(panos[0]))
    print(f"candidate panoramas: {len(candidates)} (after excluding test scenes)")

    # 3) Sample fixed pool
    rng = random.Random(args.seed)
    pool = sorted(rng.sample(candidates, min(args.pool_size, len(candidates))))
    print(f"pool size: {len(pool)}")

    pool_path = out_dir / "noise_pool.json"
    json.dump({"pool_size": len(pool), "seed": args.seed, "pool": pool}, open(pool_path, "w"), indent=2)
    print(f"wrote {pool_path}")

    # 4) Per-sample deterministic assignment (hash of sample_id -> pool index)
    assignments = {}
    for s in samples:
        h = int(hashlib.sha256(s["sample_id"].encode()).hexdigest(), 16)
        assignments[s["sample_id"]] = pool[h % len(pool)]

    assign_path = out_dir / "noise_assignments.json"
    json.dump(assignments, open(assign_path, "w"), indent=2)
    print(f"wrote {assign_path} ({len(assignments)} assignments)")

    # 5) Sanity: verify zero overlap with test scene_ids
    leaked = []
    for sample_id, pano_path in assignments.items():
        scene = Path(pano_path).parent.name
        if scene in test_scene_ids:
            leaked.append((sample_id, scene))
    if leaked:
        print(f"ERROR: {len(leaked)} leaked assignments")
        raise SystemExit(1)
    print("zero leak: OK")


if __name__ == "__main__":
    main()
