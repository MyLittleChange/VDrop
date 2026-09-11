#!/usr/bin/env python3
"""Plot answer-token attention distributions per layer across model runs.

Default inputs compare:
  - vanilla BAGEL with forced bridge generation
  - visual-only LoRA
  - bridge-mask LoRA

The script reads attention_distribution_wide.csv files produced by the attention
probe batch scripts and draws per-layer mean curves with p25-p75 bands.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


DEFAULT_VANILLA_ROOT = Path(
    "/path/to/scratch/VisualCoT/artifacts/vanilla_force_bridge_attention_100"
)
DEFAULT_BRIDGE_ROOT = Path(
    "/path/to/scratch/VisualCoT/artifacts/bridge_mask_attention_100"
)

MODEL_SPECS = [
    {
        "key": "vanilla_force_bridge",
        "run": "vanilla_bagel_force_bridge",
        "label": "Vanilla + force bridge",
        "root_arg": "vanilla_root",
    },
    {
        "key": "visual_only_lora_7k",
        "run": "visual_only_lora_7k",
        "label": "Visual-only LoRA 7K",
        "root_arg": "bridge_root",
    },
    {
        "key": "bridge_mask_lora_7k",
        "run": "bridge_mask_lora_7k",
        "label": "Bridge-mask LoRA 7K",
        "root_arg": "bridge_root",
    },
]

BATCH_MODEL_SPECS = [
    {
        "key": "vanilla_force_bridge",
        "run": "vanilla_force_bridge",
        "label": "Vanilla + force bridge",
        "root_arg": "batch_root",
    },
    {
        "key": "visual_only_lora_7k",
        "run": "visual_only_lora_7k",
        "label": "Visual-only LoRA 7K",
        "root_arg": "batch_root",
    },
    {
        "key": "bridge_mask_lora_7k",
        "run": "bridge_mask_lora_7k",
        "label": "Bridge-mask LoRA 7K",
        "root_arg": "batch_root",
    },
]

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
]

COVERED_RELATIVE_GROUPS = [
    "V1",
    "V2",
    "bridge_all",
    "text",
    "answer_prefix",
    "current_query",
]

PLOT_GROUPS = {
    "covered_parts": COVERED_RELATIVE_GROUPS,
    "bridge": ["bridge_all", "bridge_vit", "bridge_vae"],
    "inputs_text": ["V1", "V2", "text", "answer_prefix", "current_query", "other_unnamed"],
    "coverage": ["named_total", "covered_total", "other_unnamed"],
}

COLORS = {
    "vanilla_force_bridge": "#4c78a8",
    "visual_only_lora_7k": "#f58518",
    "bridge_mask_lora_7k": "#54a24b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw per-layer answer-token attention distributions for three BAGEL runs.",
    )
    parser.add_argument("--vanilla_root", type=Path, default=DEFAULT_VANILLA_ROOT)
    parser.add_argument("--bridge_root", type=Path, default=DEFAULT_BRIDGE_ROOT)
    parser.add_argument(
        "--batch_root",
        type=Path,
        default=None,
        help="Single batch output root containing attention_distribution_wide.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to <batch_root>/attention_layer_compare if --batch_root is set; otherwise <bridge_root>/../attention_layer_compare_100.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Attention groups to include in the summary and plots. Defaults depend on normalization mode.",
    )
    parser.add_argument(
        "--covered_relative",
        action="store_true",
        help="Plot each selected group divided by covered_total instead of total attention.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--no_individual",
        action="store_true",
        help="Skip one PNG per attention group.",
    )
    return parser.parse_args()


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "plot"


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def read_wide_rows(
    root: Path,
    wanted_run: str,
    model_key: str,
    model_label: str,
    groups: Sequence[str],
    covered_relative: bool = False,
) -> List[Dict[str, object]]:
    path = root / "attention_distribution_wide.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing attention wide CSV: {path}")

    rows: List[Dict[str, object]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("run") != wanted_run:
                continue
            layer = int(row["layer"])
            sample_index = int(row["sample_index"])
            denominator = None
            if covered_relative:
                denominator_value = row.get("covered_total", "")
                if denominator_value == "":
                    continue
                denominator = float(denominator_value)
                if denominator <= 0:
                    continue
            for group in groups:
                value = row.get(group, "")
                if value == "":
                    continue
                raw_value = float(value)
                attention_mass = raw_value / denominator if covered_relative else raw_value
                rows.append({
                    "model_key": model_key,
                    "model": model_label,
                    "run": wanted_run,
                    "checkpoint": row.get("checkpoint", ""),
                    "sample_index": sample_index,
                    "sample_id": row.get("sample_id", ""),
                    "layer": layer,
                    "group": group,
                    "attention_mass": attention_mass,
                    "raw_attention_mass": raw_value,
                    "normalization_denominator": denominator if covered_relative else "",
                    "predicted_answer": row.get("predicted_answer", ""),
                    "target_step_source": row.get("target_step_source", ""),
                    "step": row.get("step", ""),
                })
    if not rows:
        raise ValueError(f"No rows found for run={wanted_run!r} in {path}")
    return rows


def write_long_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = [
        "model_key",
        "model",
        "run",
        "checkpoint",
        "sample_index",
        "sample_id",
        "layer",
        "group",
        "attention_mass",
        "raw_attention_mass",
        "normalization_denominator",
        "predicted_answer",
        "target_step_source",
        "step",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    values: Dict[Tuple[str, str, str, int], List[float]] = defaultdict(list)
    labels: Dict[str, str] = {}
    for row in rows:
        model_key = str(row["model_key"])
        labels[model_key] = str(row["model"])
        key = (model_key, str(row["model"]), str(row["group"]), int(row["layer"]))
        values[key].append(float(row["attention_mass"]))

    summary: List[Dict[str, object]] = []
    for (model_key, model, group, layer), vals in sorted(values.items(), key=lambda item: (item[0][2], item[0][3], item[0][0])):
        arr = np.asarray(vals, dtype=np.float64)
        summary.append({
            "model_key": model_key,
            "model": model,
            "group": group,
            "layer": layer,
            "n": len(vals),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(vals) > 1 else 0.0,
            "sem": float(arr.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            "median": float(np.median(arr)),
            "p25": percentile(vals, 25),
            "p75": percentile(vals, 75),
        })
    return summary


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_key", "model", "group", "layer", "n", "mean", "std", "sem", "median", "p25", "p75"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_for(summary_rows: Sequence[Dict[str, object]], model_key: str, group: str) -> List[Dict[str, object]]:
    rows = [
        row for row in summary_rows
        if row["model_key"] == model_key and row["group"] == group
    ]
    return sorted(rows, key=lambda row: int(row["layer"]))


def plot_group_grid(
    summary_rows: Sequence[Dict[str, object]],
    model_specs: Sequence[Dict[str, str]],
    groups: Sequence[str],
    output_path: Path,
    title: str,
    y_label: str,
    dpi: int,
    ncols: int = 3,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [group for group in groups if any(row["group"] == group for row in summary_rows)]
    if not groups:
        return

    ncols = min(ncols, len(groups))
    nrows = math.ceil(len(groups) / ncols)
    fig_width = max(8.0, 5.2 * ncols)
    fig_height = max(4.4, 3.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.flatten()

    model_keys = [spec["key"] for spec in model_specs]
    model_labels = {spec["key"]: spec["label"] for spec in model_specs}

    for ax, group in zip(axes_flat, groups):
        for model_key in model_keys:
            model_rows = rows_for(summary_rows, model_key, group)
            if not model_rows:
                continue
            xs = np.asarray([int(row["layer"]) for row in model_rows], dtype=np.int64)
            mean = np.asarray([float(row["mean"]) * 100 for row in model_rows], dtype=np.float64)
            p25 = np.asarray([float(row["p25"]) * 100 for row in model_rows], dtype=np.float64)
            p75 = np.asarray([float(row["p75"]) * 100 for row in model_rows], dtype=np.float64)
            color = COLORS.get(model_key)
            ax.plot(xs, mean, label=model_labels[model_key], color=color, linewidth=2)
            ax.fill_between(xs, p25, p75, color=color, alpha=0.15, linewidth=0)
        ax.set_title(group)
        ax.set_xlabel("Layer")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(left=0)

    for ax in axes_flat[len(groups):]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        legend_cols = min(3, len(labels))
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=legend_cols,
            frameon=False,
        )
    fig.suptitle(title, y=0.98, fontsize=14)
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_readme(
    path: Path,
    args: argparse.Namespace,
    model_specs: Sequence[Dict[str, str]],
    output_files: Sequence[Path],
) -> None:
    data = {
        "batch_root": str(args.batch_root) if args.batch_root else None,
        "vanilla_root": str(args.vanilla_root),
        "bridge_root": str(args.bridge_root),
        "covered_relative": args.covered_relative,
        "groups": args.groups,
        "output_files": [str(path) for path in output_files],
        "models": model_specs,
    }
    metric_text = (
        "Mean lines show each selected group's share of covered attention "
        "(group / covered_total) per layer. "
        if args.covered_relative
        else "Mean lines show the average answer-token attention mass per layer. "
    )
    path.write_text(
        "# Attention Layer Comparison\n\n"
        + metric_text
        + "Shaded bands show p25-p75 across samples.\n\n"
        + "```json\n"
        + json.dumps(data, indent=2)
        + "\n```\n"
    )


def main() -> None:
    args = parse_args()
    if args.groups is None:
        args.groups = COVERED_RELATIVE_GROUPS if args.covered_relative else DEFAULT_GROUPS
    if args.output_dir is None:
        if args.batch_root is not None:
            dirname = "attention_layer_compare_covered_relative" if args.covered_relative else "attention_layer_compare"
            args.output_dir = args.batch_root / dirname
        else:
            dirname = "attention_layer_compare_covered_relative" if args.covered_relative else "attention_layer_compare_100"
            args.output_dir = args.bridge_root.parent / dirname
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = args.output_dir / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    if args.batch_root is not None:
        model_specs = BATCH_MODEL_SPECS
        roots = {
            "batch_root": args.batch_root,
        }
    else:
        model_specs = MODEL_SPECS
        roots = {
            "vanilla_root": args.vanilla_root,
            "bridge_root": args.bridge_root,
        }
    all_rows: List[Dict[str, object]] = []
    for spec in model_specs:
        all_rows.extend(
            read_wide_rows(
                root=roots[spec["root_arg"]],
                wanted_run=spec["run"],
                model_key=spec["key"],
                model_label=spec["label"],
                groups=args.groups,
                covered_relative=args.covered_relative,
            )
        )

    long_path = args.output_dir / "attention_layer_compare_long.csv"
    summary_path = args.output_dir / "attention_layer_compare_summary.csv"
    write_long_csv(long_path, all_rows)
    summary_rows = summarize(all_rows)
    write_summary_csv(summary_path, summary_rows)

    output_files = [long_path, summary_path]
    grid_path = args.output_dir / "attention_layer_compare_grid.png"
    metric_title = (
        "Answer-token attention share within covered tokens by layer"
        if args.covered_relative
        else "Answer-token attention by layer"
    )
    y_label = "Share of covered attention (%)" if args.covered_relative else "Attention mass (%)"
    plot_group_grid(
        summary_rows,
        model_specs,
        args.groups,
        grid_path,
        metric_title,
        y_label,
        dpi=args.dpi,
    )
    output_files.append(grid_path)

    for name, groups in PLOT_GROUPS.items():
        selected = [group for group in groups if group in args.groups]
        if not selected:
            continue
        path = args.output_dir / f"attention_layer_compare_{name}.png"
        plot_group_grid(
            summary_rows,
            model_specs,
            selected,
            path,
            f"{metric_title}: {name}",
            y_label,
            dpi=args.dpi,
            ncols=min(3, len(selected)),
        )
        output_files.append(path)

    if not args.no_individual:
        individual_dir = args.output_dir / "groups"
        for group in args.groups:
            path = individual_dir / f"{sanitize_filename(group)}.png"
            plot_group_grid(
                summary_rows,
                model_specs,
                [group],
                path,
                f"{metric_title}: {group}",
                y_label,
                dpi=args.dpi,
                ncols=1,
            )
            output_files.append(path)

    readme_path = args.output_dir / "README.md"
    write_readme(readme_path, args, model_specs, output_files)
    output_files.append(readme_path)

    print("Wrote:")
    for path in output_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
