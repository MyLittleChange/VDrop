#!/usr/bin/env python
"""Emit per-subcategory MMSI and MindCube markdown tables for PROGRESS.md.

Reads `evaluated_model_results_run_1.json` (MMSI judged output) and
`merged_results.json` (MindCube merge_mindcube_results.py output) for the
8 rows under PROGRESS.md §2 Setting A "+ LoRA SFT (r=32, α=64), 7K samples
balanced" plus the vanilla BAGEL row.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

ROWS: list[tuple[str, Optional[str], Optional[str]]] = [
    (
        "BAGEL (vanilla)",
        None,
        "/path/to/scratch/mindcube/BAGEL/merged_results.json",
    ),
    (
        "No Thinking",
        "/path/to/scratch/VisualCoT/BAGEL_format_training_data_mix_all_balance_no_thinking_map_lora_7k_mmsi_eval/evaluated_model_results_run_1.json",
        "/path/to/scratch/mindcube/BAGEL_format_training_data_mix_all_balance_no_thinking_map_lora_7k/merged_results.json",
    ),
    (
        "Text Think",
        "/path/to/scratch/VisualCoT/mmsi_results/BAGEL_format_training_data_mix_all_balance_text_thinking_lora_mmsi/evaluated_model_results_run_1.json",
        "/path/to/scratch/mindcube/BAGEL_format_training_data_mix_all_balance_text_thinking_lora_7k/merged_results.json",
    ),
    (
        "Panoramic View",
        "/path/to/scratch/VisualCoT/mmsi_results/BAGEL_format_training_data_mix_all_balance_visual_only_lora_mmsi/evaluated_model_results_run_1.json",
        "/path/to/scratch/mindcube/BAGEL_format_training_data_mix_all_balance_visual_only_lora_7k/merged_results.json",
    ),
    (
        "Panoramic View (Bridge-Masked)",
        "/path/to/scratch/VisualCoT/eval_outputs/BAGEL_format_mix_all_balance_visual_only_bridge_masked_lora_6k_mmsi/evaluated_model_results_run_1.json",
        "/path/to/scratch/VisualCoT/eval_outputs/BAGEL_format_mix_all_balance_visual_only_bridge_masked_lora_6k_mindcube/merged_results.json",
    ),
    (
        "Top-down View",
        "/path/to/scratch/VisualCoT/mmsi_results/BAGEL_format_training_data_mix_all_balance_topdown_visual_only_topdown_lora_mmsi/evaluated_model_results_run_1.json",
        None,
    ),
    (
        "Point Matching",
        "/path/to/scratch/VisualCoT/eval_outputs/BAGEL_format_training_data_pm_no_rotation_visual_only_lora_mmsi/evaluated_model_results_run_1.json",
        "/path/to/scratch/VisualCoT/eval_outputs/BAGEL_format_training_data_pm_no_rotation_visual_only_lora_mindcube/merged_results.json",
    ),
    (
        "Top-down Real Env",
        "/path/to/scratch/VisualCoT/eval_outputs/BAGEL_format_training_data_topdown_round3_visual_only_lora_mmsi/evaluated_model_results_run_1.json",
        "/path/to/scratch/VisualCoT/eval_outputs/BAGEL_format_training_data_topdown_round3_visual_only_lora_mindcube/merged_results.json",
    ),
]

MMSI_TYPES = [
    "Attribute (Appr.)",
    "Attribute (Meas.)",
    "MSR",
    "Motion (Cam.)",
    "Motion (Obj.)",
    "Positional Relationship (Cam.–Cam.)",
    "Positional Relationship (Cam.–Obj.)",
    "Positional Relationship (Cam.–Reg.)",
    "Positional Relationship (Obj.–Obj.)",
    "Positional Relationship (Obj.–Reg.)",
    "Positional Relationship (Reg.–Reg.)",
]

MMSI_SHORT = {
    "Attribute (Appr.)": "Attr (Appr)",
    "Attribute (Meas.)": "Attr (Meas)",
    "MSR": "MSR",
    "Motion (Cam.)": "Mot (Cam)",
    "Motion (Obj.)": "Mot (Obj)",
    "Positional Relationship (Cam.–Cam.)": "Pos (C–C)",
    "Positional Relationship (Cam.–Obj.)": "Pos (C–O)",
    "Positional Relationship (Cam.–Reg.)": "Pos (C–R)",
    "Positional Relationship (Obj.–Obj.)": "Pos (O–O)",
    "Positional Relationship (Obj.–Reg.)": "Pos (O–R)",
    "Positional Relationship (Reg.–Reg.)": "Pos (R–R)",
}

MINDCUBE_CAT0 = ["linear", "perpendicular"]
MINDCUBE_CAT1 = ["O-O", "OO", "P-O", "P-P", "PO"]


def fmt(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.1f}"


def mmsi_breakdown(path: str) -> dict[str, Optional[float]]:
    """Return {QuestionType: accuracy_fraction, "Overall": acc}."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of records in {path}")

    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
    for r in data:
        qt = r.get("QuestionType", "unknown")
        stats[qt][1] += 1
        if r.get("LLMJudgeResult") is True:
            stats[qt][0] += 1

    out: dict[str, Optional[float]] = {}
    for t in MMSI_TYPES:
        c, n = stats.get(t, [0, 0])
        out[t] = (c / n) if n > 0 else None

    total = sum(s[1] for s in stats.values())
    correct = sum(s[0] for s in stats.values())
    out["Overall"] = correct / total if total > 0 else None
    return out


def mindcube_breakdown(path: str) -> dict[str, Optional[float]]:
    with open(path) as f:
        data = json.load(f)
    m = data["metrics"]
    out: dict[str, Optional[float]] = {}
    cat0 = m.get("by_category_0", {})
    cat1 = m.get("by_category_1", {})
    for k in MINDCUBE_CAT0:
        out[k] = cat0.get(k)
    for k in MINDCUBE_CAT1:
        out[k] = cat1.get(k)
    out["Overall"] = m.get("overall_accuracy")
    return out


def render_table(header: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(header[i]), max(len(r[i]) for r in rows)) for i in range(len(header))
    ]
    fmt_row = lambda cells: "| " + " | ".join(
        cells[i].ljust(widths[i]) for i in range(len(cells))
    ) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    lines = [fmt_row(header), sep]
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)


def main() -> None:
    # MMSI table
    mmsi_header = ["Method"] + [MMSI_SHORT[t] for t in MMSI_TYPES] + ["Overall"]
    mmsi_rows = []
    for label, mmsi_path, _ in ROWS:
        if mmsi_path is None or not Path(mmsi_path).is_file():
            cells = [label] + ["—"] * (len(MMSI_TYPES) + 1)
        else:
            br = mmsi_breakdown(mmsi_path)
            cells = [label] + [fmt(br[t]) for t in MMSI_TYPES] + [fmt(br["Overall"])]
        mmsi_rows.append(cells)

    print("### MMSI-Bench per-question-type (Setting A · LoRA SFT 7K balanced)\n")
    print(render_table(mmsi_header, mmsi_rows))
    print()

    # MindCube table
    mc_header = (
        ["Method"] + MINDCUBE_CAT0 + MINDCUBE_CAT1 + ["Overall"]
    )
    mc_rows = []
    for label, _, mc_path in ROWS:
        if mc_path is None or not Path(mc_path).is_file():
            cells = [label] + ["—"] * (len(MINDCUBE_CAT0) + len(MINDCUBE_CAT1) + 1)
        else:
            br = mindcube_breakdown(mc_path)
            cells = (
                [label]
                + [fmt(br[k]) for k in MINDCUBE_CAT0]
                + [fmt(br[k]) for k in MINDCUBE_CAT1]
                + [fmt(br["Overall"])]
            )
        mc_rows.append(cells)

    print("### MindCube per-category (Setting A · LoRA SFT 7K balanced)\n")
    print(render_table(mc_header, mc_rows))


if __name__ == "__main__":
    main()
