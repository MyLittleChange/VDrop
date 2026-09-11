#!/usr/bin/env python3
"""
Generates a JSONL of remaining (not-yet-completed) VSI-Bench samples,
given completed results in existing shard/checkpoint files.

Usage:
    python prepare_resume_dataset.py \
        --output_dir /path/to/scratch/vsibench/BAGEL_format_topdown_qa_visual_only \
        --data_dir /path/to/scratch/datasets/VSI-Bench \
        --dataset_file test_debiased.parquet
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
parser.add_argument("--data_dir", required=True)
parser.add_argument("--dataset_file", default="test_debiased.parquet")
parser.add_argument("--shard0_file", default="inference_results_shard0.json")
parser.add_argument("--checkpoint_file", default="inference_results_shard1_checkpoint.json")
parser.add_argument("--out_file", default="remaining_samples.jsonl")
args = parser.parse_args()

completed_ids = set()
for fname in [args.shard0_file, args.checkpoint_file]:
    fpath = os.path.join(args.output_dir, fname)
    if not os.path.exists(fpath):
        print(f"  (skipping missing: {fpath})")
        continue
    with open(fpath) as f:
        data = json.load(f)
    results = data.get("results", data) if isinstance(data, dict) else data
    for r in results:
        if r.get("final_answer_text", "").strip():
            completed_ids.add(str(r["sample_id"]))
print(f"Completed sample IDs: {len(completed_ids)}")

dataset_path = os.path.join(args.data_dir, args.dataset_file)
df = pd.read_parquet(dataset_path)
all_data = df.to_dict("records")
print(f"Total dataset size: {len(all_data)}")

remaining = [r for r in all_data if str(r.get("id", "")) not in completed_ids]
print(f"Remaining samples: {len(remaining)}")

out_path = os.path.join(args.output_dir, args.out_file)
with open(out_path, "w") as f:
    for row in remaining:
        f.write(json.dumps(row, cls=_NumpyEncoder) + "\n")
print(f"Written to: {out_path}")
