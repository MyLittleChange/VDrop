#!/usr/bin/env python3
"""Draw stacked bars for covered-relative answer-token attention parts.

This reads attention_layer_compare_summary.csv from the covered-relative plot
directory. Each bar is one model at one selected layer, and the stack segments
sum the covered attention parts:
  V1 + V2, bridge_all, text, and answer_prefix + current_query.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


DEFAULT_ROOT = Path(
    "/path/to/scratch/VisualCoT/artifacts/blink_multiview_attention_val133"
)
DEFAULT_SUMMARY = (
    DEFAULT_ROOT
    / "attention_layer_compare_covered_relative"
    / "attention_layer_compare_summary.csv"
)
DEFAULT_OUTPUT = (
    DEFAULT_ROOT
    / "attention_layer_compare_covered_relative"
    / "attention_layer_compare_covered_parts.png"
)

MODEL_ORDER = [
    ("vanilla_force_bridge", "Vanilla"),
    ("visual_only_lora_7k", "Visual-only"),
    ("bridge_mask_lora_7k", "Bridge-mask"),
]

PART_SPECS = [
    ("V1 + V2", ("V1", "V2"), "#8BCF8B", "/"),
    ("bridge_all", ("bridge_all",), "#E9A6A1", "\\"),
    ("text", ("text",), "#FFF6CC", ""),
    ("answer_prefix + current_query", ("answer_prefix", "current_query"), "#3775BA", "x"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stack covered-relative attention parts for selected layers.",
    )
    parser.add_argument("--summary_csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", type=int, nargs="+", default=[5, 11, 17, 23, 27])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--title",
        default="Covered answer-token attention parts at selected layers",
    )
    return parser.parse_args()


def read_summary(
    path: Path,
) -> Tuple[Dict[Tuple[str, int, str], float], Dict[str, str]]:
    means: Dict[Tuple[str, int, str], float] = {}
    labels: Dict[str, str] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_key = row["model_key"]
            labels[model_key] = row["model"]
            means[(model_key, int(row["layer"]), row["group"])] = float(row["mean"])
    return means, labels


def get_part_value(
    means: Dict[Tuple[str, int, str], float],
    model_key: str,
    layer: int,
    groups: Iterable[str],
) -> float:
    return sum(means.get((model_key, layer, group), 0.0) for group in groups)


def validate_inputs(
    means: Dict[Tuple[str, int, str], float],
    layers: Sequence[int],
) -> None:
    missing: List[str] = []
    for model_key, _ in MODEL_ORDER:
        for layer in layers:
            for _, groups, _, _ in PART_SPECS:
                for group in groups:
                    if (model_key, layer, group) not in means:
                        missing.append(f"{model_key} layer={layer} group={group}")
    if missing:
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise ValueError(f"Missing summary rows:\n{preview}{suffix}")


def plot_stacked_layers(
    means: Dict[Tuple[str, int, str], float],
    layers: Sequence[int],
    output_path: Path,
    title: str,
    dpi: int,
) -> List[Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = output_path.parent / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["hatch.linewidth"] = 0.8

    n_layers = len(layers)
    n_models = len(MODEL_ORDER)
    bar_width = 0.23
    cluster_spacing = 1.35
    centers = np.arange(n_layers, dtype=np.float64) * cluster_spacing
    offsets = (np.arange(n_models, dtype=np.float64) - (n_models - 1) / 2.0) * bar_width

    fig_width = max(10.0, 2.25 * n_layers)
    fig, ax = plt.subplots(figsize=(fig_width, 6.2))

    bar_positions: List[float] = []
    bar_labels: List[str] = []
    for layer_idx, layer in enumerate(layers):
        for model_idx, (model_key, model_label) in enumerate(MODEL_ORDER):
            x = centers[layer_idx] + offsets[model_idx]
            bottom = 0.0
            for part_label, groups, color, hatch in PART_SPECS:
                height = 100.0 * get_part_value(means, model_key, layer, groups)
                ax.bar(
                    x,
                    height,
                    bar_width * 0.92,
                    bottom=bottom,
                    color=color,
                    edgecolor="black",
                    linewidth=1.1,
                    hatch=hatch,
                    label=part_label if layer_idx == 0 and model_idx == 0 else None,
                )
                bottom += height
            bar_positions.append(x)
            bar_labels.append(model_label)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of covered attention (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, rotation=35, ha="right")
    ax.tick_params(axis="x", length=0)

    for center, layer in zip(centers, layers):
        ax.text(
            center,
            -0.17,
            f"Layer {layer}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    for separator in (centers[:-1] + centers[1:]) / 2.0:
        ax.axvline(separator, color="#dddddd", linewidth=0.8, zorder=0)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.21),
        ncol=len(PART_SPECS),
        frameon=False,
        columnspacing=1.2,
        handlelength=1.6,
    )

    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    output_paths = [output_path]
    if output_path.suffix.lower() != ".pdf":
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        output_paths.append(pdf_path)
    plt.close(fig)
    return output_paths


def main() -> None:
    args = parse_args()
    means, _ = read_summary(args.summary_csv)
    validate_inputs(means, args.layers)
    output_paths = plot_stacked_layers(
        means=means,
        layers=args.layers,
        output_path=args.output_path,
        title=args.title,
        dpi=args.dpi,
    )
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
