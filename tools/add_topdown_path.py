#!/usr/bin/env python3
"""Add topdown_path field to relative dataset JSON.

For each sample, determines which agent has the question (user_1_question
or user_2_question is non-null) and links the corresponding pre-generated
topdown map PNG.
"""

import json
import os

DATASET = "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json"
TOPDOWN_BASE = "/path/to/scratch/VisualCoT/infinigen/rendered_panorama_relative_distance_normalized"

OLD_PREFIX = "/path/to/scratch"
NEW_PREFIX = "/path/to/scratch"

with open(DATASET) as f:
    raw = f.read().replace(OLD_PREFIX, NEW_PREFIX)
    data = json.loads(raw)

found, missing, no_question = 0, 0, 0

for sample in data:
    scene_id = sample["scene_id"]
    sample_id = sample["sample_id"]


    path = f"{TOPDOWN_BASE}/{scene_id}/panorama_blender_limits_{sample_id}.png"
    if os.path.exists(path):
        sample["panorama_path"] = path
        found += 1
    else:
        sample["panorama_path"] = None
        missing += 1

# Write back
with open(DATASET, "w") as f:
    json.dump(data, f, indent=2)

print(f"Done: {found} found, {missing} missing, {no_question} no question, {len(data)} total")
