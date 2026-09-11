#!/usr/bin/env python3
"""Aggregate per-sample answer attention CSVs into batch summaries."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_GROUPS = [
    "V1",
    "V2",
    "bridge_vae",
    "bridge_vit",
    "bridge_all",
    "text",
    "answer_prefix",
    "current_query",
    "named_total",
    "covered_total",
    "other_unnamed",
    "total",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate answer-token attention groups across batch probe outputs.",
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run_names", nargs="+", required=True)
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument("--sample_limit", type=int, default=None)
    return parser.parse_args()


def pctile(values: Sequence[float], q: float) -> float | str:
    values = sorted(values)
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float | str) -> str:
    if value == "":
        return ""
    return f"{100 * float(value):.2f}%"


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    manifest_samples = list(manifest["samples"])
    if args.sample_limit is not None:
        manifest_samples = manifest_samples[:args.sample_limit]
    sample_lookup = {
        int(item["sample_index"]): item
        for item in manifest_samples
    }

    wide_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for run_name in args.run_names:
        run_root = output_root / "runs" / run_name
        for sample_index, sample in sample_lookup.items():
            sample_dir = run_root / f"sample_{sample_index}"
            meta_path = sample_dir / "metadata.json"
            groups_path = sample_dir / "attention_groups.csv"
            if not meta_path.exists() or not groups_path.exists():
                failures.append({
                    "run": run_name,
                    "sample_index": sample_index,
                    "sample_id": sample.get("sample_id", ""),
                    "reason": "missing metadata or attention_groups",
                })
                continue
            meta = json.loads(meta_path.read_text())
            keyed: Dict[tuple[int, int], Dict[str, float]] = {}
            with groups_path.open("r", newline="") as f:
                for row in csv.DictReader(f):
                    key = (int(row["step"]), int(row["layer"]))
                    keyed.setdefault(key, {})
                    keyed[key][row["group"]] = float(row["attention_mass"])

            for (step, layer), values in keyed.items():
                out: Dict[str, Any] = {
                    "run": run_name,
                    "checkpoint": meta.get("model_name", ""),
                    "sample_index": sample_index,
                    "sample_id": meta.get("sample_id", sample.get("sample_id", "")),
                    "dataset_kind": meta.get("dataset_kind", manifest.get("dataset_kind", "")),
                    "question_type": meta.get("question_type", sample.get("sub_task", "")),
                    "sub_task": meta.get("sub_task", sample.get("sub_task", "")),
                    "predicted_answer": meta.get("predicted_answer"),
                    "correct_answer_idx": meta.get("correct_answer_idx"),
                    "correct_answer": meta.get("correct_answer"),
                    "target_step_source": meta.get("target_step_source", ""),
                    "step": step,
                    "layer": layer,
                }
                for group in args.groups:
                    out[group] = values.get(group, "")
                wide_rows.append(out)

    wide_path = output_root / "attention_distribution_wide.csv"
    write_csv(wide_path, wide_rows)

    summary_rows: List[Dict[str, Any]] = []
    for run_name in args.run_names:
        run_rows = [row for row in wide_rows if row["run"] == run_name]
        layers = sorted({row["layer"] for row in run_rows})
        for layer in layers:
            layer_rows = [row for row in run_rows if row["layer"] == layer]
            for group in args.groups:
                vals = [row[group] for row in layer_rows if row[group] != ""]
                if not vals:
                    continue
                summary_rows.append({
                    "run": run_name,
                    "layer": layer,
                    "group": group,
                    "n": len(vals),
                    "mean": statistics.mean(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "median": statistics.median(vals),
                    "p25": pctile(vals, 0.25),
                    "p75": pctile(vals, 0.75),
                })
    summary_path = output_root / "attention_distribution_summary.csv"
    write_csv(summary_path, summary_rows)

    baseline = args.run_names[0]
    delta_rows: List[Dict[str, Any]] = []
    base_map = {
        (row["sample_index"], row["layer"], group): row[group]
        for row in wide_rows if row["run"] == baseline
        for group in args.groups if row[group] != ""
    }
    for candidate in args.run_names[1:]:
        cand_map = {
            (row["sample_index"], row["layer"], group): row[group]
            for row in wide_rows if row["run"] == candidate
            for group in args.groups if row[group] != ""
        }
        for key in sorted(set(base_map) & set(cand_map)):
            sample_index, layer, group = key
            delta_rows.append({
                "baseline": baseline,
                "candidate": candidate,
                "sample_index": sample_index,
                "sample_id": sample_lookup.get(sample_index, {}).get("sample_id", ""),
                "layer": layer,
                "group": group,
                "baseline_attention": base_map[key],
                "candidate_attention": cand_map[key],
                "delta_candidate_minus_baseline": cand_map[key] - base_map[key],
            })
    delta_path = output_root / "attention_distribution_paired_deltas.csv"
    write_csv(delta_path, delta_rows)

    delta_summary_rows: List[Dict[str, Any]] = []
    for candidate in args.run_names[1:]:
        candidate_rows = [row for row in delta_rows if row["candidate"] == candidate]
        for layer in sorted({row["layer"] for row in candidate_rows}):
            for group in args.groups:
                vals = [
                    row["delta_candidate_minus_baseline"]
                    for row in candidate_rows
                    if row["layer"] == layer and row["group"] == group
                ]
                if not vals:
                    continue
                delta_summary_rows.append({
                    "baseline": baseline,
                    "candidate": candidate,
                    "layer": layer,
                    "group": group,
                    "n": len(vals),
                    "mean_delta": statistics.mean(vals),
                    "std_delta": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "median_delta": statistics.median(vals),
                    "p25_delta": pctile(vals, 0.25),
                    "p75_delta": pctile(vals, 0.75),
                })
    delta_summary_path = output_root / "attention_distribution_delta_summary.csv"
    write_csv(delta_summary_path, delta_summary_rows)

    fail_path = output_root / "attention_distribution_failures.csv"
    write_csv(fail_path, failures)

    summary_lookup = {
        (row["run"], row["layer"], row["group"]): row
        for row in summary_rows
    }
    md_lines = [
        "# Attention Distribution Summary",
        "",
        f"Dataset kind: `{manifest.get('dataset_kind', '')}`",
        f"Data dir: `{manifest.get('data_dir', manifest.get('dataset_json', ''))}`",
        f"Task/split: `{manifest.get('task', '')}` / `{manifest.get('split', '')}`",
        f"Requested samples: `{len(manifest_samples)}`",
        f"Runs: `{', '.join(args.run_names)}`",
        "",
    ]
    if summary_rows:
        md_lines.append("| layer | group | " + " | ".join(f"{run} mean" for run in args.run_names) + " |")
        md_lines.append("|---:|:---|" + "|".join("---:" for _ in args.run_names) + "|")
        layers = sorted({row["layer"] for row in summary_rows})
        for layer in layers:
            for group in args.groups:
                values = [
                    fmt_pct(summary_lookup.get((run, layer, group), {}).get("mean", ""))
                    for run in args.run_names
                ]
                if not any(values):
                    continue
                md_lines.append(f"| {layer} | `{group}` | " + " | ".join(values) + " |")

    if failures:
        md_lines.extend(["", f"Missing/failed rows: `{len(failures)}`. See `{fail_path}`."])

    md_path = output_root / "attention_distribution_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"Wrote {wide_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {delta_path}")
    print(f"Wrote {delta_summary_path}")
    print(f"Wrote {fail_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
