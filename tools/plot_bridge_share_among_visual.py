#!/usr/bin/env python3
"""Plot bridge share among visual attention tokens.

The metric is:

    bridge_all / (V1 + V2 + bridge_all)

This removes text, answer-prefix, current-query, and unnamed delimiter tokens
from the denominator, so it asks: among named visual sources, how much goes to
the generated bridge?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


DEFAULT_BATCH_ROOT = Path(
    "/path/to/scratch/VisualCoT/artifacts/blink_multiview_attention_val133"
)

RUN_LABELS = {
    "vanilla_force_bridge": "BAGEL (Vanilla)",
    "visual_only_lora_7k": "Visual Thinking w/o VDrop",
    "bridge_mask_lora_7k": "Visual Thinking w/ VDrop",
}

COLORS = {
    "vanilla_force_bridge": "#4c78a8",
    "visual_only_lora_7k": "#f58518",
    "bridge_mask_lora_7k": "#54a24b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot bridge_all / (V1 + V2 + bridge_all) by layer.",
    )
    parser.add_argument("--batch_root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--run_names",
        nargs="+",
        default=["vanilla_force_bridge", "visual_only_lora_7k", "bridge_mask_lora_7k"],
    )
    parser.add_argument("--ymin", type=float, default=0.0, help="Y-axis minimum in percent.")
    parser.add_argument("--ymax", type=float, default=100.0, help="Y-axis maximum in percent.")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def pctile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_long_rows(path: Path, run_names: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            run = row["run"]
            if run not in run_names:
                continue
            v1 = float(row["V1"])
            v2 = float(row["V2"])
            bridge = float(row["bridge_all"])
            visual_total = v1 + v2 + bridge
            if visual_total <= 0:
                continue
            rows.append({
                "run": run,
                "model": RUN_LABELS.get(run, run),
                "checkpoint": row.get("checkpoint", ""),
                "sample_index": int(row["sample_index"]),
                "sample_id": row.get("sample_id", ""),
                "layer": int(row["layer"]),
                "V1": v1,
                "V2": v2,
                "bridge_all": bridge,
                "visual_total": visual_total,
                "V1_share_among_visual": v1 / visual_total,
                "V2_share_among_visual": v2 / visual_total,
                "bridge_share_among_visual": bridge / visual_total,
            })
    return rows


def summarize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    values: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in rows:
        values[(row["run"], int(row["layer"]))].append(float(row["bridge_share_among_visual"]))

    out: List[Dict[str, Any]] = []
    for (run, layer), vals in sorted(values.items()):
        arr = np.asarray(vals, dtype=np.float64)
        out.append({
            "run": run,
            "model": RUN_LABELS.get(run, run),
            "layer": layer,
            "n": len(vals),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(vals) > 1 else 0.0,
            "median": float(np.median(arr)),
            "p25": pctile(vals, 25),
            "p75": pctile(vals, 75),
        })
    return out


def summarize_ranges(summary_rows: Sequence[Dict[str, Any]], run_names: Sequence[str]) -> List[Dict[str, Any]]:
    ranges = [
        ("all_layers", range(0, 28)),
        ("late_half", range(14, 28)),
        ("last_quarter", range(21, 28)),
        ("last4", range(24, 28)),
    ]
    lookup = {
        (row["run"], int(row["layer"])): float(row["mean"])
        for row in summary_rows
    }
    out: List[Dict[str, Any]] = []
    for run in run_names:
        for range_name, layer_range in ranges:
            vals = [
                lookup[(run, layer)]
                for layer in layer_range
                if (run, layer) in lookup
            ]
            if not vals:
                continue
            out.append({
                "run": run,
                "model": RUN_LABELS.get(run, run),
                "layer_range": range_name,
                "mean_bridge_share_among_visual": float(np.mean(vals)),
            })
    return out


def plot_by_layer(
    summary_rows: Sequence[Dict[str, Any]],
    run_names: Sequence[str],
    output_path: Path,
    dpi: int,
    ymin: float,
    ymax: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.4))
    for run in run_names:
        rows = sorted(
            [row for row in summary_rows if row["run"] == run],
            key=lambda row: int(row["layer"]),
        )
        if not rows:
            continue
        xs = np.asarray([int(row["layer"]) for row in rows], dtype=np.int64)
        mean = np.asarray([float(row["mean"]) * 100 for row in rows], dtype=np.float64)
        p25 = np.asarray([float(row["p25"]) * 100 for row in rows], dtype=np.float64)
        p75 = np.asarray([float(row["p75"]) * 100 for row in rows], dtype=np.float64)
        ax.plot(xs, mean, label=RUN_LABELS.get(run, run), color=COLORS.get(run), linewidth=2)
        ax.fill_between(xs, p25, p75, color=COLORS.get(run), alpha=0.15, linewidth=0)

    ax.set_xlabel("Layer", fontsize=16)
    ax.set_ylabel("Attention share on thinking image\n(among visual tokens, %)", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_xlim(left=0)
    ax.set_ylim(ymin, ymax)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=False, fontsize=15)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(left=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def write_readme(path: Path, args: argparse.Namespace, range_rows: Sequence[Dict[str, Any]], output_files: Sequence[Path]) -> None:
    lines = [
        "# Bridge Share Among Visual Attention",
        "",
        "Metric:",
        "",
        "`bridge_all / (V1 + V2 + bridge_all)`",
        "",
        "This excludes text, answer-prefix, current-query, and unnamed delimiter tokens.",
        "",
        "| run | all layers | late half | last quarter | last 4 |",
        "|:---|---:|---:|---:|---:|",
    ]
    lookup = {
        (row["run"], row["layer_range"]): float(row["mean_bridge_share_among_visual"])
        for row in range_rows
    }
    for run in args.run_names:
        vals = [
            lookup.get((run, "all_layers")),
            lookup.get((run, "late_half")),
            lookup.get((run, "last_quarter")),
            lookup.get((run, "last4")),
        ]
        lines.append(
            f"| `{run}` | "
            + " | ".join("" if val is None else f"{100 * val:.2f}%" for val in vals)
            + " |"
        )
    lines.extend([
        "",
        "Files:",
        "",
    ])
    for file_path in output_files:
        lines.append(f"- `{file_path}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.batch_root / "bridge_share_among_visual")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    (output_dir / ".matplotlib").mkdir(parents=True, exist_ok=True)

    wide_path = args.batch_root / "attention_distribution_wide.csv"
    long_rows = load_long_rows(wide_path, args.run_names)
    summary_rows = summarize(long_rows)
    range_rows = summarize_ranges(summary_rows, args.run_names)

    long_path = output_dir / "bridge_share_among_visual_long.csv"
    summary_path = output_dir / "bridge_share_among_visual_by_layer.csv"
    ranges_path = output_dir / "bridge_share_among_visual_ranges.csv"
    png_path = output_dir / "bridge_share_among_visual_by_layer.png"
    pdf_path = png_path.with_suffix(".pdf")
    readme_path = output_dir / "README.md"

    write_csv(long_path, long_rows)
    write_csv(summary_path, summary_rows)
    write_csv(ranges_path, range_rows)
    plot_by_layer(summary_rows, args.run_names, png_path, args.dpi, args.ymin, args.ymax)
    write_readme(readme_path, args, range_rows, [long_path, summary_path, ranges_path, png_path, pdf_path])

    print("Wrote:")
    for path in [long_path, summary_path, ranges_path, png_path, pdf_path, readme_path]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
