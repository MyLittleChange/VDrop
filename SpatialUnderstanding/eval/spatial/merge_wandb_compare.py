"""
Merge two BAGEL eval runs over the same samples into a single shareable W&B
comparison table.

Run A's data lives in W&B (table artifact + images bundled).
Run B is local (inference_results_bagel_merged.json + generated_images/).

Joins on sample_id, builds a wandb.Table with side-by-side image columns +
predicted answers + correctness, logs to a new W&B run.

Example:
    python merge_wandb_compare.py \\
        --run-a-artifact aishwarya-agrawal-mila-org/bagel-spatial-reasoning/run-p4wyro51-inference_results:v0 \\
        --run-b-dir /path/to/scratch/VisualCoT/BAGEL_format_mix_anchor_counting_distance_balance_visual_only_lora_6k_mcqs_relative_direction_normalized \\
        --run-a-name mix_all_rotation_visual_only \\
        --run-b-name mix_anchor_counting_distance_lora_6k \\
        --out-project bagel-spatial-reasoning \\
        --out-entity aishwarya-agrawal-mila-org
"""

import argparse
import json
import os
import sys
from pathlib import Path

import wandb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-a-artifact", required=True,
                   help="W&B artifact path of Run A's run_table, e.g. "
                        "entity/project/run-<id>-inference_results:v0")
    p.add_argument("--run-b-dir", required=True, type=Path,
                   help="Local directory containing inference_results_bagel_merged.json "
                        "and generated_images/")
    p.add_argument("--source-dataset", type=Path,
                   default=Path("/path/to/scratch/VisualCoT/spatial_collab_dataset/"
                                "approved_mcqs_relative_direction_normalized.json"),
                   help="Source MCQ dataset JSON used to look up input images per sample_id")
    p.add_argument("--data-dir", type=Path,
                   default=Path("/path/to/scratch/VisualCoT/spatial_collab_dataset"),
                   help="Data root used for relative image paths in the source dataset")
    p.add_argument("--run-a-name", default="run_a")
    p.add_argument("--run-b-name", default="run_b")
    p.add_argument("--run-a-description", default="",
                   help="Short description of Run A (e.g. training-set composition); "
                        "used in the W&B run notes/description")
    p.add_argument("--run-b-description", default="",
                   help="Short description of Run B")
    p.add_argument("--out-project", required=True)
    p.add_argument("--out-entity", default=None)
    p.add_argument("--out-run-name", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="If set, only process the first N samples (smoke test)")
    return p.parse_args()


def load_run_a(artifact_path: str):
    api = wandb.Api()
    art = api.artifact(artifact_path, type="run_table")
    local_dir = art.download()
    table_json = Path(local_dir) / "inference_results.table.json"
    with open(table_json) as f:
        t = json.load(f)
    cols = t["columns"]
    rows = []
    for row in t["data"]:
        d = dict(zip(cols, row))
        img_path = None
        if isinstance(d.get("generated_image"), dict):
            rel = d["generated_image"].get("path")
            if rel:
                img_path = str(Path(local_dir) / rel)
        d["_image_path"] = img_path
        rows.append(d)
    return rows, local_dir


def resolve_input_image_path(raw: str, data_dir: Path) -> str:
    if not isinstance(raw, str) or not raw:
        return ""
    if raw.startswith("images/"):
        return str(data_dir / raw)
    return raw.replace(
        "/path/to/scratch", "/path/to/scratch"
    )


def load_source_inputs(source_path: Path, data_dir: Path) -> dict:
    with open(source_path) as f:
        items = json.load(f)
    by_id = {}
    for it in items:
        sid = it.get("sample_id")
        if not sid:
            continue
        by_id[sid] = {
            "input_image_1": resolve_input_image_path(it.get("user_1_image_local_path", ""), data_dir),
            "input_image_2": resolve_input_image_path(it.get("user_2_image_local_path", ""), data_dir),
        }
    return by_id


def load_run_b(run_b_dir: Path):
    json_path = run_b_dir / "inference_results_bagel_merged.json"
    with open(json_path) as f:
        payload = json.load(f)
    by_id = {}
    for r in payload["results"]:
        by_id[r["sample_id"]] = {
            "predicted_answer": r["predicted_answer"],
            "correct_answer": r["correct_answer"],
            "correct": bool(r.get("accuracy", 0.0) >= 1.0),
            "image_path": r["saved_image_paths"][0] if r.get("saved_image_paths") else None,
            "question": r.get("question", ""),
            "options": r.get("options", []),
        }
    return by_id, payload.get("metrics", {})


def agreement_label(a_correct: bool, b_correct: bool) -> str:
    if a_correct and b_correct:
        return "both_correct"
    if not a_correct and not b_correct:
        return "both_wrong"
    if a_correct and not b_correct:
        return "a_only"
    return "b_only"


def main():
    args = parse_args()

    print(f"Loading Run A artifact {args.run_a_artifact}")
    a_rows, a_local_dir = load_run_a(args.run_a_artifact)
    if args.limit:
        a_rows = a_rows[: args.limit]
    a_correct_total = sum(1 for r in a_rows if bool(r.get("correct")))
    print(f"  {len(a_rows)} rows; accuracy = {a_correct_total / max(len(a_rows), 1):.4f}")
    print(f"  artifact files at {a_local_dir}")

    print(f"Loading source dataset for input images: {args.source_dataset}")
    inputs_by_id = load_source_inputs(args.source_dataset, args.data_dir)
    print(f"  {len(inputs_by_id)} samples in source dataset")

    print(f"Loading Run B from {args.run_b_dir}")
    run_b_by_id, run_b_metrics = load_run_b(args.run_b_dir)
    print(f"  {len(run_b_by_id)} rows; reported overall_accuracy = "
          f"{run_b_metrics.get('overall_accuracy')}")

    a_by_id = {r["sample_id"]: r for r in a_rows}
    a_ids = set(a_by_id.keys())
    b_ids = set(run_b_by_id.keys())
    common = sorted(a_ids & b_ids)
    print(f"Join: |A|={len(a_ids)} |B|={len(b_ids)} |intersection|={len(common)}")
    only_a = a_ids - b_ids
    only_b = b_ids - a_ids
    if only_a:
        print(f"  [warn] {len(only_a)} only in A: {sorted(only_a)[:5]}...")
    if only_b:
        print(f"  [warn] {len(only_b)} only in B: {sorted(only_b)[:5]}...")

    out_run_name = args.out_run_name or f"compare_{args.run_a_name}_vs_{args.run_b_name}"
    print(f"Initializing W&B run: project={args.out_project} name={out_run_name}")
    wandb.init(
        project=args.out_project,
        entity=args.out_entity,
        name=out_run_name,
        job_type="comparison",
        config={
            "run_a_artifact": args.run_a_artifact,
            "run_a_name": args.run_a_name,
            "run_b_dir": str(args.run_b_dir),
            "run_b_name": args.run_b_name,
            "n_samples": len(common),
        },
    )

    a_col = args.run_a_name
    b_col = args.run_b_name
    columns = [
        "sample_id", "question", "options", "correct_answer",
        "input_image_1", "input_image_2",
        f"{a_col}_pred", f"{a_col}_correct", f"{a_col}_image",
        f"{b_col}_pred", f"{b_col}_correct", f"{b_col}_image",
        "agreement",
    ]
    table = wandb.Table(columns=columns)

    counts = {"both_correct": 0, "both_wrong": 0, "a_only": 0, "b_only": 0}
    a_correct_n = 0
    b_correct_n = 0
    skipped = 0
    missing_input = 0

    for sid in common:
        a = a_by_id[sid]
        b = run_b_by_id[sid]
        src = inputs_by_id.get(sid, {})

        a_img_path = a.get("_image_path")
        b_img_path = b["image_path"]
        a_img = wandb.Image(a_img_path) if a_img_path and os.path.exists(a_img_path) else None
        b_img = wandb.Image(b_img_path) if b_img_path and os.path.exists(b_img_path) else None
        if a_img is None or b_img is None:
            skipped += 1

        in1 = src.get("input_image_1", "")
        in2 = src.get("input_image_2", "")
        in1_img = wandb.Image(in1) if in1 and os.path.exists(in1) else None
        in2_img = wandb.Image(in2) if in2 and os.path.exists(in2) else None
        if in1_img is None or in2_img is None:
            missing_input += 1

        a_corr = bool(a.get("correct"))
        b_corr = bool(b["correct"])
        ag = agreement_label(a_corr, b_corr)
        counts[ag] += 1
        a_correct_n += int(a_corr)
        b_correct_n += int(b_corr)

        table.add_data(
            sid,
            a.get("question", ""),
            a.get("options", ""),
            a.get("correct_answer", ""),
            in1_img, in2_img,
            a.get("predicted_answer", ""), a_corr, a_img,
            b["predicted_answer"], b_corr, b_img,
            ag,
        )

    n = len(common)
    summary_rows = [
        ("metric", "value"),
        (f"{a_col}_accuracy", a_correct_n / n if n else 0.0),
        (f"{b_col}_accuracy", b_correct_n / n if n else 0.0),
        ("n_samples", float(n)),
        ("both_correct", float(counts["both_correct"])),
        ("both_wrong", float(counts["both_wrong"])),
        ("a_only_correct", float(counts["a_only"])),
        ("b_only_correct", float(counts["b_only"])),
        ("missing_image_rows", float(skipped)),
        ("missing_input_image_rows", float(missing_input)),
    ]
    summary_table = wandb.Table(columns=list(summary_rows[0]),
                                data=[list(r) for r in summary_rows[1:]])

    wandb.log({"comparison_table": table, "summary": summary_table})
    if n:
        wandb.summary[f"{a_col}_accuracy"] = a_correct_n / n
        wandb.summary[f"{b_col}_accuracy"] = b_correct_n / n
    wandb.summary["n_samples"] = n
    wandb.summary.update({f"agreement_{k}": v for k, v in counts.items()})

    print("Summary:")
    for k, v in summary_rows[1:]:
        print(f"  {k}: {v}")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
