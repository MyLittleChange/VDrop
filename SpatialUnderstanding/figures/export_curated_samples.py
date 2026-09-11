#!/usr/bin/env python3
"""
Export curated case-study samples to a flat directory structure.

Output layout (one subdir per sample_id):

    <output_root>/<sample_id>/
        question.txt
        input_cam0.png
        input_cam1.png
        panoramic_pred-<X>_<correct|wrong>.png
        point_matching_pred-<X>_<correct|wrong>.png
        corner_view_pred-<X>_<correct|wrong>.png

Each generated image is the BAGEL output that the model produced at inference
time (not the GT oracle). question.txt is plain text: question + options +
gold + a 4-line per-bridge accuracy summary.

Usage:
    python SpatialUnderstanding/figures/export_curated_samples.py
"""

import argparse
import glob
import json
import os
import shutil


EVAL_ROOT = "/path/to/scratch/VisualCoT"
DATASET_ROOT = "/path/to/scratch/VisualCoT/spatial_collab_dataset"

# Per-subtask BAGEL output-dir suffixes; full dir = "<bridge_prefix><suffix>".
SUBTASK_SUFFIX = {
    "anchor":        "mcqs_anchor_normalized",
    "counting":      "mcqs_counting_normalized",
    "rel-distance":  "mcqs_relative_distance_normalized",
    "rel-direction": "mcqs_relative_direction_normalized",
}
BRIDGE_PREFIX = {
    "panoramic":      "BAGEL_format_mix_all_balance_visual_only_bridge_masked_lora_7k_",
    "point_matching": "BAGEL_format_training_data_pm_no_rotation_visual_only_lora_7k_",
    "corner_view":    "BAGEL_format_corner_view_visual_only_bridge_masked_lora_7k_",
}
DATASET_FILES = {
    "anchor":        "approved_mcqs_anchor_normalized.json",
    "counting":      "approved_mcqs_counting_normalized.json",
    "rel-distance":  "approved_mcqs_relative_distance_normalized.json",
    "rel-direction": "approved_mcqs_relative_direction_normalized.json",
}

# (subtask, sample_id) — verified to exist.
CURATED = [
    ("anchor",        "anchor_001147"),
    ("counting",      "counting_003053"),
    ("rel-distance",  "relative_distance_008495"),
    ("rel-direction", "spatial_008738"),
]


def load_eval_rows(subdir):
    rows = []
    for sh in sorted(glob.glob(os.path.join(EVAL_ROOT, subdir, "inference_results_bagel_shard*.json"))):
        rows.extend(json.load(open(sh)).get("results", []))
    return {r["sample_id"]: r for r in rows}


def load_dataset_row(subtask, sample_id):
    path = os.path.join(DATASET_ROOT, DATASET_FILES[subtask])
    for r in json.load(open(path)):
        if r["sample_id"] == sample_id:
            return r
    return None


def export_sample(subtask, sample_id, output_root):
    sample_dir = os.path.join(output_root, sample_id)
    os.makedirs(sample_dir, exist_ok=True)
    print(f"\n→ {sample_id}  ({subtask}) → {sample_dir}")

    ds_row = load_dataset_row(subtask, sample_id)
    if ds_row is None:
        print(f"   [SKIP] sample not in dataset {DATASET_FILES[subtask]}")
        return

    # 1) question.txt — Q + options + gold + per-bridge result.
    question = ds_row.get("question_both_views") or ds_row.get("question") or ""
    options = ds_row.get("options_user_2") or ds_row.get("options_user_1") or ds_row.get("options") or []
    gold = ds_row.get("correct_answer", "")
    letters = "ABCDEFGHIJ"

    # Gather per-bridge results before writing question.txt so we can include
    # them in the summary.
    per_bridge_results = {}
    for bridge, prefix in BRIDGE_PREFIX.items():
        subdir = prefix + SUBTASK_SUFFIX[subtask]
        rows = load_eval_rows(subdir)
        r = rows.get(sample_id)
        if r is None:
            per_bridge_results[bridge] = None
            continue
        is_correct = float(r.get("accuracy", 0.0)) >= 0.5
        per_bridge_results[bridge] = {
            "pred": r.get("predicted_answer", ""),
            "is_correct": is_correct,
            "image_path": r.get("saved_image_paths", [None])[0],
        }

    lines = [
        f"sample_id: {sample_id}",
        f"subtask:   {subtask}",
        "",
        "Question:",
        f"  {question}",
        "",
        "Options:",
    ]
    for i, opt in enumerate(options):
        lines.append(f"  {letters[i] if i < len(letters) else i}. {opt}")
    lines += ["", f"Gold answer: {gold}", "", "BAGEL per-bridge predictions:"]
    for bridge, res in per_bridge_results.items():
        if res is None:
            lines.append(f"  {bridge:<14} : (not found)")
        else:
            mark = "✓" if res["is_correct"] else "✗"
            lines.append(f"  {bridge:<14} : pred={res['pred']!r}  {mark}")
    qfile = os.path.join(sample_dir, "question.txt")
    with open(qfile, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"   wrote question.txt")

    # 2) input views — copy cam0 and cam1 from the dataset row.
    cam_paths = [
        ("input_cam0.png", ds_row.get("user_1_image_local_path")),
        ("input_cam1.png", ds_row.get("user_2_image_local_path")),
    ]
    for fname, src in cam_paths:
        if not src or not os.path.exists(src):
            print(f"   [WARN] {fname} source missing: {src}")
            continue
        dst = os.path.join(sample_dir, fname)
        shutil.copyfile(src, dst)
        print(f"   copied {fname}  ({os.path.getsize(dst)//1024} KB)")

    # 3) BAGEL generated images, one per bridge, named with pred + correct/wrong.
    for bridge, res in per_bridge_results.items():
        if res is None:
            print(f"   [WARN] {bridge}: row not found, skipping")
            continue
        src = res["image_path"]
        if not src or not os.path.exists(src):
            print(f"   [WARN] {bridge}: generated image missing at {src}")
            continue
        mark = "correct" if res["is_correct"] else "wrong"
        pred = res["pred"] or "NA"
        dst_name = f"{bridge}_pred-{pred}_{mark}.png"
        dst = os.path.join(sample_dir, dst_name)
        shutil.copyfile(src, dst)
        print(f"   copied {dst_name}  ({os.path.getsize(dst)//1024} KB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output_root",
        default=os.path.expanduser("~/scratch/VisualCoT/case_studies/curated_4samples"),
    )
    args = p.parse_args()

    print(f"Exporting {len(CURATED)} curated samples → {args.output_root}")
    os.makedirs(args.output_root, exist_ok=True)

    for subtask, sample_id in CURATED:
        export_sample(subtask, sample_id, args.output_root)

    print(f"\nDone. Tree:")
    os.system(f"ls -la {args.output_root}")
    for subtask, sid in CURATED:
        d = os.path.join(args.output_root, sid)
        if os.path.isdir(d):
            print(f"\n{d}:")
            os.system(f"ls -la {d}")


if __name__ == "__main__":
    main()
