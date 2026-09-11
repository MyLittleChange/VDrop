#!/usr/bin/env python3
"""Upload BAGEL eval results (MindCube / spatial MCQ / MMSI) to Weights & Biases.

Auto-detects the eval type per directory and opens one W&B run per dir.
"""

import argparse
import json
import os
import re
from typing import Iterable

import wandb


SPATIAL_DATASET_ROOT = "/path/to/scratch/VisualCoT/spatial_collab_dataset"
SPATIAL_DATASET_FILES = {
    "spatial_mcq_anchor": "approved_mcqs_anchor_normalized.json",
    "spatial_mcq_counting": "approved_mcqs_counting_normalized.json",
    "spatial_mcq_relative_direction": "approved_mcqs_relative_direction_normalized.json",
    "spatial_mcq_relative_distance": "approved_mcqs_relative_distance_normalized.json",
    "spatial_mcq_map": "approved_dataset_map_questions_normalized.json",
}

# MMSI shared image bank — the eval JSON embeds Tamia paths that don't resolve on Mila.
MMSI_IMAGE_ROOT = "/path/to/scratch/VisualCoT/BAGEL_format_training_data_mix_balance_matterport_visual_only_lora_mmsi_eval/images"


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def build_id_to_images(dataset_file: str) -> dict:
    id_to_images = {}
    if not os.path.exists(dataset_file):
        print(f"Warning: dataset file not found: {dataset_file}")
        return id_to_images
    with open(dataset_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            id_to_images[entry["id"]] = entry.get("images", [])
    return id_to_images


def build_spatial_id_to_paths(dataset_file: str) -> dict:
    """Build sample_id → {user_1, user_2, topdown} for spatial MCQ datasets."""
    out = {}
    if not os.path.exists(dataset_file):
        print(f"Warning: spatial dataset file not found: {dataset_file}")
        return out
    data = load_json(dataset_file)
    for entry in data:
        sid = entry.get("sample_id")
        if sid:
            out[sid] = {
                "user_1": entry.get("user_1_image_local_path"),
                "user_2": entry.get("user_2_image_local_path"),
                "topdown": entry.get("topdown_path") or entry.get("center_view_image_path"),
                "panorama": entry.get("panorama_path"),
            }
    return out


def remap_mmsi_path(p: str, eval_dir: str) -> str:
    """Translate stale Tamia MMSI paths to the on-cluster equivalents."""
    if not p:
        return p
    # Generated images: …/<some_run>_mmsi/pass_1/task_<id>/generated_round_*.png
    m = re.search(r"/pass_1/(task_\d+/generated_round_\d+\.png)$", p)
    if m:
        return os.path.join(eval_dir, "pass_1", m.group(1))
    # Input images: …/<some_run>_mmsi/images/<id>_<n>.jpg → shared image bank
    m = re.search(r"/images/(\d+_\d+\.jpe?g)$", p)
    if m:
        return os.path.join(MMSI_IMAGE_ROOT, m.group(1))
    return p


def _safe_image(path: str):
    if not path:
        return None
    if not os.path.exists(path):
        print(f"Warning: image not found: {path}")
        return None
    try:
        return wandb.Image(path)
    except Exception as e:
        print(f"Warning: could not load image {path}: {e}")
        return None


def _flatten_metrics(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_metrics(v, key))
        elif isinstance(v, list):
            # Skip lists of heterogeneous content; keep short numeric lists as-is.
            if all(isinstance(x, (int, float)) for x in v):
                out[key] = v
        else:
            out[key] = v
    return out


def detect_eval_type(d: str) -> str | None:
    """Return 'mindcube', 'spatial_mcq', 'mmsi', or None."""
    merged = os.path.join(d, "merged_results.json")
    if os.path.exists(merged):
        try:
            data = load_json(merged)
            if "by_category_0" in data.get("metrics", {}):
                return "mindcube"
        except Exception as e:
            print(f"Warning: could not parse {merged}: {e}")

    mcq = os.path.join(d, "inference_results_bagel_merged.json")
    if os.path.exists(mcq):
        try:
            data = load_json(mcq)
            if "per_type" in data.get("metrics", {}):
                return "spatial_mcq"
        except Exception as e:
            print(f"Warning: could not parse {mcq}: {e}")

    summary = os.path.join(d, "summary.json")
    evaluated = os.path.join(d, "evaluated_model_results_run_1.json")
    if os.path.exists(summary) and os.path.exists(evaluated):
        return "mmsi"

    return None


VARIANT_TAG_STRIP = "BAGEL_format_training_data_mix_all_"


def _short_variant_tag(variant: str) -> str:
    """Return a W&B-tag-safe (<=64 char) short form of the variant."""
    short = variant[len(VARIANT_TAG_STRIP):] if variant.startswith(VARIANT_TAG_STRIP) else variant
    tag = f"variant:{short}"
    if len(tag) > 64:
        tag = tag[:64]
    return tag


def parse_variant_and_benchmark(dir_basename: str, eval_type: str) -> tuple[str, str]:
    """Extract the variant root and a benchmark label for W&B tagging."""
    if eval_type == "mindcube":
        return dir_basename, "mindcube"
    if eval_type == "mmsi":
        variant = dir_basename[: -len("_mmsi_eval")] if dir_basename.endswith("_mmsi_eval") else dir_basename
        return variant, "mmsi"
    if eval_type == "spatial_mcq":
        m = re.match(r"^(.+?)_mcqs_([a-z_]+)_normalized$", dir_basename)
        if m:
            return m.group(1), f"spatial_mcq_{m.group(2)}"
        if dir_basename.endswith("_map"):
            return dir_basename[: -len("_map")], "spatial_mcq_map"
        return dir_basename, "spatial_mcq"
    return dir_basename, eval_type


def upload_mindcube(d: str, args) -> None:
    results_file = os.path.join(d, "merged_results.json")
    data = load_json(results_file)
    metrics = data.get("metrics", {})
    config = data.get("config", {})
    results = data.get("results", [])
    if args.max_samples is not None:
        results = results[: args.max_samples]

    run_name = os.path.basename(d.rstrip("/"))
    variant, benchmark = parse_variant_and_benchmark(run_name, "mindcube")

    dataset_path = os.path.join(args.mindcube_data_dir, args.mindcube_dataset_file)
    print(f"[mindcube] Pre-loading dataset from {dataset_path}")
    id_to_images = build_id_to_images(dataset_path)

    print(f"[mindcube] Initializing W&B run: {run_name}")
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        tags=[_short_variant_tag(variant), f"benchmark:{benchmark}"],
        config={
            "results_file": results_file,
            "eval_type": "mindcube",
            "variant": variant,
            "benchmark": benchmark,
            "mindcube_data_dir": args.mindcube_data_dir,
            "mindcube_dataset_file": args.mindcube_dataset_file,
            **config,
        },
        reinit=True,
    )

    flat = _flatten_metrics(metrics)
    print(f"[mindcube] Logging {len(flat)} flattened metrics")
    wandb.log(flat)

    columns = [
        "sample_id",
        "question",
        "input_images",
        "generated_image",
        "predicted_answer",
        "gt_answer",
        "correct",
        "answer_text",
        "category",
    ]
    table = wandb.Table(columns=columns)
    for r in results:
        sid = r.get("sample_id", "")
        input_imgs = []
        for rel in id_to_images.get(sid, []):
            img = _safe_image(os.path.join(args.mindcube_data_dir, rel))
            if img is not None:
                input_imgs.append(img)
        saved = r.get("saved_image_paths", [])
        gen = _safe_image(saved[0]) if saved else None
        table.add_data(
            sid,
            r.get("question", ""),
            input_imgs if input_imgs else None,
            gen,
            r.get("predicted_answer", ""),
            r.get("gt_answer", ""),
            r.get("accuracy", 0.0) > 0.5,
            r.get("final_answer_text", ""),
            " | ".join(r.get("category", [])),
        )

    print(f"[mindcube] Uploading table with {len(table.data)} rows")
    wandb.log({"inference_results": table})
    wandb.finish()
    print(f"[mindcube] Done: {run.url}")


def upload_spatial_mcq(d: str, args) -> None:
    results_file = os.path.join(d, "inference_results_bagel_merged.json")
    data = load_json(results_file)
    metrics = data.get("metrics", {})
    config = data.get("config", {})
    results = data.get("results", [])
    if args.max_samples is not None:
        results = results[: args.max_samples]

    run_name = os.path.basename(d.rstrip("/"))
    variant, benchmark = parse_variant_and_benchmark(run_name, "spatial_mcq")

    dataset_file = SPATIAL_DATASET_FILES.get(benchmark)
    id_to_paths = {}
    if dataset_file:
        path = os.path.join(SPATIAL_DATASET_ROOT, dataset_file)
        print(f"[spatial_mcq] Pre-loading dataset from {path}")
        id_to_paths = build_spatial_id_to_paths(path)
    else:
        print(f"[spatial_mcq] Warning: no dataset mapping for benchmark={benchmark}; input images will be empty")

    print(f"[spatial_mcq] Initializing W&B run: {run_name}")
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        tags=[_short_variant_tag(variant), f"benchmark:{benchmark}"],
        config={
            "results_file": results_file,
            "eval_type": "spatial_mcq",
            "variant": variant,
            "benchmark": benchmark,
            "dataset_file": dataset_file,
            **config,
        },
        reinit=True,
    )

    flat = _flatten_metrics(metrics)
    print(f"[spatial_mcq] Logging {len(flat)} flattened metrics")
    wandb.log(flat)

    columns = [
        "sample_id",
        "question_type",
        "input_image_1",
        "input_image_2",
        "topdown",
        "generated_image",
        "question",
        "options",
        "predicted_answer",
        "correct_answer",
        "correct_answer_idx",
        "correct",
        "answer_text",
    ]
    table = wandb.Table(columns=columns)
    for r in results:
        saved = r.get("saved_image_paths", [])
        gen = _safe_image(saved[0]) if saved else None
        paths = id_to_paths.get(r.get("sample_id", ""), {})
        img1 = _safe_image(paths.get("user_1"))
        img2 = _safe_image(paths.get("user_2"))
        topdown = _safe_image(paths.get("topdown"))
        table.add_data(
            r.get("sample_id", ""),
            r.get("question_type", ""),
            img1,
            img2,
            topdown,
            gen,
            r.get("question", ""),
            " | ".join(r.get("options", [])),
            r.get("predicted_answer", ""),
            r.get("correct_answer", ""),
            r.get("correct_answer_idx", -1),
            r.get("accuracy", 0.0) > 0.5,
            r.get("final_answer_text", ""),
        )

    print(f"[spatial_mcq] Uploading table with {len(table.data)} rows")
    wandb.log({"inference_results": table})
    wandb.finish()
    print(f"[spatial_mcq] Done: {run.url}")


def upload_mmsi(d: str, args) -> None:
    summary = load_json(os.path.join(d, "summary.json"))
    results = load_json(os.path.join(d, "evaluated_model_results_run_1.json"))
    if args.max_samples is not None:
        results = results[: args.max_samples]

    # Derive per-type / per-difficulty accuracies from the count pairs.
    derived = {}
    for block_key in ("per_type", "per_difficulty"):
        block = summary.get(block_key, {})
        for label, stats in block.items():
            total = stats.get("total", 0) or 0
            correct = stats.get("correct", 0) or 0
            if total > 0:
                derived[f"{block_key}.{label}.accuracy"] = correct / total

    metrics = {
        "overall_accuracy": summary.get("overall_accuracy"),
        "total_samples": summary.get("total_samples"),
        **_flatten_metrics({"per_type": summary.get("per_type", {})}),
        **_flatten_metrics({"per_difficulty": summary.get("per_difficulty", {})}),
        **derived,
    }
    missing = summary.get("missing_shards", [])

    run_name = os.path.basename(d.rstrip("/"))
    variant, benchmark = parse_variant_and_benchmark(run_name, "mmsi")

    print(f"[mmsi] Initializing W&B run: {run_name}")
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        tags=[_short_variant_tag(variant), f"benchmark:{benchmark}"],
        config={
            "eval_dir": d,
            "eval_type": "mmsi",
            "variant": variant,
            "benchmark": benchmark,
            "missing_shards": missing,
            "num_missing_shards": len(missing),
        },
        reinit=True,
    )

    if missing:
        print(f"[mmsi] Warning: {len(missing)} missing shards in summary — eval is partial")

    print(f"[mmsi] Logging {len(metrics)} metrics")
    wandb.log(metrics)

    columns = [
        "sample_id",
        "question_type",
        "difficulty",
        "question",
        "input_images",
        "generated_image",
        "extracted_answer",
        "gt_answer",
        "correct",
        "thought",
        "answer_text",
    ]
    table = wandb.Table(columns=columns)
    for r in results:
        input_imgs = [
            img
            for img in (_safe_image(remap_mmsi_path(p, d)) for p in r.get("ImagePaths", []))
            if img is not None
        ]
        gen_paths = [remap_mmsi_path(p, d) for p in r.get("GeneratedImages", [])]
        gen = _safe_image(gen_paths[0]) if gen_paths else None
        table.add_data(
            r.get("Id", ""),
            r.get("QuestionType", ""),
            r.get("Difficulty", ""),
            r.get("OriginalQuestion", r.get("Question", "")),
            input_imgs if input_imgs else None,
            gen,
            r.get("ExtractedAnswer", ""),
            r.get("GroundTruth", ""),
            bool(r.get("LLMJudgeResult", False)),
            r.get("Thought", ""),
            r.get("ModelResult", ""),
        )

    print(f"[mmsi] Uploading table with {len(table.data)} rows")
    wandb.log({"inference_results": table})
    wandb.finish()
    print(f"[mmsi] Done: {run.url}")


HANDLERS = {
    "mindcube": upload_mindcube,
    "spatial_mcq": upload_spatial_mcq,
    "mmsi": upload_mmsi,
}


# --- Compare mode --------------------------------------------------------------------

BLINK_PARQUET_DEFAULT = "/path/to/scratch/datasets/BLINK/Multi-view_Reasoning/val-00000-of-00001.parquet"


def _load_blink_input_images(parquet_path: str) -> dict:
    """Return sample_id -> (PIL.Image, PIL.Image) for BLINK Multi-view_Reasoning."""
    import io
    import pandas as pd
    from PIL import Image

    if not os.path.exists(parquet_path):
        print(f"Warning: BLINK parquet not found: {parquet_path}")
        return {}
    df = pd.read_parquet(parquet_path)
    out = {}
    for _, row in df.iterrows():
        sid = row["idx"]
        try:
            img1 = Image.open(io.BytesIO(row["image_1"]["bytes"])).convert("RGB") if row["image_1"] is not None else None
            img2 = Image.open(io.BytesIO(row["image_2"]["bytes"])).convert("RGB") if row["image_2"] is not None else None
            out[sid] = (img1, img2)
        except Exception as e:
            print(f"Warning: could not decode BLINK images for {sid}: {e}")
    return out


def _resolve_saved_image(path: str, eval_dir: str) -> str:
    """Map stale Tamia generated-image paths to the local eval_dir's generated_images/."""
    if not path:
        return path
    if os.path.exists(path):
        return path
    base = os.path.basename(path)
    cand = os.path.join(eval_dir, "generated_images", base)
    if os.path.exists(cand):
        return cand
    return path


def _find_merged_json(d: str) -> str:
    """Return whichever merged JSON exists in d; raise if none."""
    for name in ("inference_results_bagel_merged.json", "merged_results.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise SystemExit(f"No merged results JSON in {d} (looked for inference_results_bagel_merged.json / merged_results.json)")


def _load_results_indexed(merged_path: str) -> tuple:
    data = load_json(merged_path)
    return (
        {r["sample_id"]: r for r in data.get("results", []) if r.get("sample_id") is not None},
        data.get("metrics", {}),
        data.get("config", {}),
    )


def _benchmark_for_dir(d: str) -> str:
    """Use the same naming logic as the per-dir uploader to label this dir."""
    eval_type = detect_eval_type(d) or "spatial_mcq"
    _, benchmark = parse_variant_and_benchmark(os.path.basename(d.rstrip("/")), eval_type)
    return benchmark


def upload_compare(dirs: list, labels: list, args) -> None:
    """Side-by-side comparison across N variants: one W&B run, one table, paired by sample_id."""
    if len(dirs) < 2:
        raise SystemExit("--compare_dirs needs at least 2 directories")
    if labels and len(labels) != len(dirs):
        raise SystemExit(f"--compare_labels must match --compare_dirs count ({len(dirs)})")
    if not labels:
        labels = [os.path.basename(d.rstrip("/")) for d in dirs]

    merged_paths = [_find_merged_json(d) for d in dirs]
    per_variant = []  # list of (label, dir, results_dict, metrics, config, benchmark)
    for d, label, mp in zip(dirs, labels, merged_paths):
        results, metrics, config = _load_results_indexed(mp)
        per_variant.append({
            "label": label,
            "dir": d,
            "results": results,
            "metrics": metrics,
            "config": config,
            "benchmark": _benchmark_for_dir(d),
        })

    id_sets = [set(v["results"]) for v in per_variant]
    common_ids = sorted(set.intersection(*id_sets))
    print(f"[compare] {len(common_ids)} shared sample_ids across {len(dirs)} variants")
    for v, ids in zip(per_variant, id_sets):
        only = ids - set(common_ids)
        print(f"[compare]   {v['label']}: {len(ids)} total, {len(only)} not shared")

    if args.max_samples is not None:
        common_ids = common_ids[: args.max_samples]

    benchmarks = {v["benchmark"] for v in per_variant}
    benchmark = next(iter(benchmarks)) if len(benchmarks) == 1 else "+".join(sorted(benchmarks))

    blink_inputs = {}
    if args.blink_parquet and any(v["benchmark"].startswith("blink") or "Multi-view_Reasoning" in str(v["config"]) for v in per_variant):
        print(f"[compare] Loading BLINK input images from {args.blink_parquet}")
        blink_inputs = _load_blink_input_images(args.blink_parquet)

    spatial_paths = {}
    if benchmark in SPATIAL_DATASET_FILES:
        ds_path = os.path.join(SPATIAL_DATASET_ROOT, SPATIAL_DATASET_FILES[benchmark])
        print(f"[compare] Loading spatial-MCQ dataset for input images: {ds_path}")
        spatial_paths = build_spatial_id_to_paths(ds_path)

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = "compare-" + "-vs-".join(labels[:3]) + ("-+more" if len(labels) > 3 else "")
        run_name = run_name[:96]
    print(f"[compare] Initializing W&B run: {run_name}")

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        tags=["compare", f"benchmark:{benchmark}"] + [_short_variant_tag(v["label"]) for v in per_variant],
        config={
            "labels": labels,
            "dirs": dirs,
            "benchmark": benchmark,
            "blink_parquet": args.blink_parquet if blink_inputs else None,
            "spatial_dataset": SPATIAL_DATASET_FILES.get(benchmark),
            "shared_samples": len(common_ids),
            "configs": {v["label"]: v["config"] for v in per_variant},
        },
        reinit=True,
    )

    # Per-variant overall + shared-set accuracy.
    summary = {"shared_samples": len(common_ids), "n_variants": len(per_variant)}
    for v in per_variant:
        summary[f"{v['label']}.overall_accuracy"] = v["metrics"].get("overall_accuracy")
        n_correct = sum(1 for sid in common_ids if v["results"][sid].get("accuracy", 0) > 0.5)
        summary[f"{v['label']}.shared_accuracy"] = n_correct / max(len(common_ids), 1)
        summary[f"{v['label']}.shared_correct"] = n_correct

    # Agreement buckets.
    all_correct = none_correct = mixed_correct = 0
    only_counts = {v["label"]: 0 for v in per_variant}
    for sid in common_ids:
        flags = [v["results"][sid].get("accuracy", 0) > 0.5 for v in per_variant]
        nc = sum(flags)
        if nc == len(per_variant):
            all_correct += 1
        elif nc == 0:
            none_correct += 1
        elif nc == 1:
            only_label = per_variant[flags.index(True)]["label"]
            only_counts[only_label] += 1
        else:
            mixed_correct += 1
    summary.update({
        "all_correct": all_correct,
        "none_correct": none_correct,
        "mixed_correct": mixed_correct,
    })
    for label, c in only_counts.items():
        summary[f"only_{label}_correct"] = c
    print(f"[compare] Logging {len(summary)} comparison metrics")
    wandb.log(summary)

    # Build table.
    base_cols = ["sample_id", "sub_task", "input_image_1", "input_image_2", "topdown", "question", "gt_answer"]
    variant_cols = []
    for v in per_variant:
        variant_cols += [f"{v['label']}_pred", f"{v['label']}_correct", f"{v['label']}_gen", f"{v['label']}_text"]
    columns = base_cols + variant_cols + ["agreement"]
    table = wandb.Table(columns=columns)

    for sid in common_ids:
        rows = [v["results"][sid] for v in per_variant]
        any_row = rows[0]

        # Input images.
        in1 = in2 = topdown_img = None
        if blink_inputs and sid in blink_inputs:
            img1, img2 = blink_inputs[sid]
            try:
                in1 = wandb.Image(img1) if img1 is not None else None
                in2 = wandb.Image(img2) if img2 is not None else None
            except Exception as e:
                print(f"Warning: could not wrap BLINK images for {sid}: {e}")
        elif spatial_paths and sid in spatial_paths:
            paths = spatial_paths[sid]
            in1 = _safe_image(paths.get("user_1"))
            in2 = _safe_image(paths.get("user_2"))
            topdown_img = _safe_image(paths.get("topdown"))

        sub_task = any_row.get("sub_task", "") or any_row.get("question_type", "")
        question = any_row.get("question", "")
        gt = any_row.get("gt_answer", "") or any_row.get("correct_answer", "")

        flags = [r.get("accuracy", 0.0) > 0.5 for r in rows]
        nc = sum(flags)
        if nc == len(rows):
            agreement = "all_correct"
        elif nc == 0:
            agreement = "none_correct"
        elif nc == 1:
            agreement = f"only_{per_variant[flags.index(True)]['label']}"
        else:
            correct_labels = [per_variant[i]["label"] for i, f in enumerate(flags) if f]
            agreement = "mixed:" + "+".join(correct_labels)

        row = [sid, sub_task, in1, in2, topdown_img, question, gt]
        for v, r in zip(per_variant, rows):
            saved = r.get("saved_image_paths", []) or []
            gen = _safe_image(_resolve_saved_image(saved[0], v["dir"])) if saved else None
            row += [
                r.get("predicted_answer", ""),
                r.get("accuracy", 0.0) > 0.5,
                gen,
                r.get("final_answer_text", ""),
            ]
        row.append(agreement)
        table.add_data(*row)

    print(f"[compare] Uploading table with {len(table.data)} rows × {len(columns)} cols")
    wandb.log({"comparison": table})
    wandb.finish()
    print(f"[compare] Done: {run.url}")


# --- CLI -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Upload BAGEL eval results to W&B (mindcube / spatial MCQ / MMSI / compare)")
    parser.add_argument("--dirs", nargs="+", help="One or more eval result directories (per-dir mode)")
    parser.add_argument("--compare_dirs", nargs="+", help="Compare 2+ eval dirs in a single W&B run/table")
    parser.add_argument("--compare_labels", nargs="+", help="Short labels for the checkpoints (default: dir basenames). Must match --compare_dirs count.")
    parser.add_argument("--run_name", default=None, help="Override W&B run name (compare mode only)")
    parser.add_argument("--blink_parquet", default=BLINK_PARQUET_DEFAULT, help="BLINK Multi-view_Reasoning parquet (for compare-mode input images)")
    parser.add_argument("--project", default="bagel-spatial-reasoning", help="W&B project")
    parser.add_argument("--entity", default=None, help="W&B entity (team or username)")
    parser.add_argument("--max_samples", type=int, default=None, help="Truncate sample table to N rows (default: all)")
    parser.add_argument("--mindcube_data_dir", default="/path/to/scratch/datasets/MindCube/data", help="Root dir for MindCube input images")
    parser.add_argument("--mindcube_dataset_file", default="raw/MindCube_tinybench.jsonl", help="MindCube dataset JSONL (relative to --mindcube_data_dir)")
    args = parser.parse_args()

    if args.compare_dirs:
        upload_compare(args.compare_dirs, args.compare_labels, args)
        return

    if not args.dirs:
        parser.error("Either --dirs or --compare_dirs is required")

    for d in args.dirs:
        d = d.rstrip("/")
        if not os.path.isdir(d):
            print(f"Skip (not a dir): {d}")
            continue
        eval_type = detect_eval_type(d)
        if eval_type is None:
            print(f"Skip (could not detect eval type): {d}")
            continue
        print(f"\n=== {d} -> {eval_type} ===")
        HANDLERS[eval_type](d, args)


if __name__ == "__main__":
    main()
