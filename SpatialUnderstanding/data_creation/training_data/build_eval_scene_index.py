#!/usr/bin/env python3
"""Build a scene_index for the 4 oracle test JSONs (738 unique scenes).

Output format matches scene_index_with_v4.json: a dict
    {"scenes": [{"scene_id", "room_part", "source_root", "user_1_image_local_path",
                 "user_2_image_local_path", "question_both_views"}, ...],
     "n_scenes": <int>}

room_part is "<version>/<room_part>" for Ankur scenes (matches what
render_topdown_blender.py expects). source_root is "ankur" for all 738 scenes.
"""
import json
from pathlib import Path

TEST_JSONS = {
    "anchor": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
    "counting": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
    "relative_distance": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
    "relative_direction": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
}
OUT_PATH = Path("/path/to/scratch/VisualCoT/training_data/map_questions_synth/scene_index_eval_oracle.json")


def main():
    seen = {}  # sid -> scene_dict
    for tag, path in TEST_JSONS.items():
        rows = json.loads(Path(path).read_text())
        for r in rows:
            sid = r["scene_id"]
            if sid in seen:
                continue
            rp = r["room_part"]
            img = r.get("user_1_image_local_path") or r.get("user_2_image_local_path") or ""
            if "/v00_filtered/" not in img:
                print(f"  SKIP {sid}: unexpected image path {img[:80]}")
                continue
            # /.../v00_filtered/<version>/<room_part>/<sid>/frames/...
            version = img.split("/v00_filtered/")[1].split("/")[0]
            full_rp = f"{version}/{rp}"
            seen[sid] = {
                "scene_id": sid,
                "room_part": full_rp,
                "asking_to": r.get("asking_to") or "agent_2",
                "user_1_image_local_path": r["user_1_image_local_path"],
                "user_2_image_local_path": r["user_2_image_local_path"],
                "source_root": "ankur",
                "question_both_views": r.get("question_both_views") or "",
            }
    scenes = sorted(seen.values(), key=lambda s: (s["room_part"], s["scene_id"]))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"scenes": scenes, "n_scenes": len(scenes)}, indent=2))
    print(f"Wrote {OUT_PATH}")
    print(f"  {len(scenes)} unique scenes")
    # Per-version histogram
    from collections import Counter
    versions = Counter(s["room_part"].split("/")[0] for s in scenes)
    for v, n in sorted(versions.items(), key=lambda x: -x[1]):
        print(f"    {v}: {n}")


if __name__ == "__main__":
    main()
