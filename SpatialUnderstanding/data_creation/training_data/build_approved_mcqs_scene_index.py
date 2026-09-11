#!/usr/bin/env python3
"""
Build a scene_index.json for the COSMIC test split, in the schema expected
by render_topdown_blender.py. One entry per unique scene_id (738), with
source_root="ankur" and room_part="<vN>/<room_part>".
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

TEST_JSONS = [
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
    "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default="/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/scene_index_approved_mcqs.json",
    )
    args = ap.parse_args()

    scenes = OrderedDict()
    for jp in TEST_JSONS:
        for x in json.load(open(jp)):
            sid = x["scene_id"]
            if sid in scenes:
                continue
            # parse <vN>/<room_part> from the user_1_image_local_path
            after = x["user_1_image_local_path"].split("/v00_filtered/", 1)[1]
            parts = after.split("/")
            version, room_part = parts[0], parts[1]
            scenes[sid] = {
                "scene_id": sid,
                "room_part": f"{version}/{room_part}",
                "source_root": "ankur",
                "user_1_image_local_path": x["user_1_image_local_path"],
                "user_2_image_local_path": x["user_2_image_local_path"],
                "asking_to": x.get("asking_to", "agent_2"),
            }

    out = {"n_scenes": len(scenes), "scenes": list(scenes.values())}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)

    versions = {}
    for s in out["scenes"]:
        v = s["room_part"].split("/")[0]
        versions[v] = versions.get(v, 0) + 1
    print(f"wrote {out_path} ({out['n_scenes']} scenes)")
    print(f"  versions: {versions}")


if __name__ == "__main__":
    main()
