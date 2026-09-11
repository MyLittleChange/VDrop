#!/usr/bin/env python3
"""Analyze which KV positions are counted as other_unnamed in attention probes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify unnamed KV positions in answer attention probe outputs.",
    )
    parser.add_argument("--batch_root", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to <batch_root>/unnamed_token_analysis.",
    )
    parser.add_argument(
        "--run_names",
        nargs="*",
        default=None,
        help="Defaults to all directories under <batch_root>/runs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pctile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return float(values[lo] * (1 - frac) + values[hi] * frac)


def choose_target_step(metadata: Dict[str, Any], token_trace: Sequence[Dict[str, Any]]) -> int:
    target_steps = metadata.get("target_steps") or []
    if target_steps:
        return int(target_steps[0])
    if not token_trace:
        raise ValueError("token_trace is empty and metadata has no target_steps")
    return int(token_trace[-1]["step"])


def covered_intervals(
    spans: Sequence[Dict[str, Any]],
    token_trace: Sequence[Dict[str, Any]],
    target_step: int,
) -> Tuple[List[Tuple[int, int, str, str]], int, int]:
    trace_by_step = {int(item["step"]): item for item in token_trace}
    if target_step not in trace_by_step:
        raise ValueError(f"target_step={target_step} is absent from token_trace")

    query_pos = int(trace_by_step[target_step]["query_kv_positions"][0])
    answer_start = int(token_trace[0]["query_kv_positions"][0])

    intervals: List[Tuple[int, int, str, str]] = []
    for span in spans:
        intervals.append((
            int(span["start"]),
            int(span["end"]),
            str(span["name"]),
            str(span.get("kind", "")),
        ))
    if query_pos > answer_start:
        intervals.append((answer_start, query_pos, "answer_prefix", "answer_prefix"))
    intervals.append((query_pos, query_pos + 1, "current_query", "current_query"))
    return sorted(intervals), answer_start, query_pos


def complement_intervals(
    covered: Sequence[Tuple[int, int, str, str]],
    end_position: int,
) -> List[Tuple[int, int]]:
    gaps: List[Tuple[int, int]] = []
    cursor = 0
    for start, end, _, _ in covered:
        if cursor < start:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < end_position:
        gaps.append((cursor, end_position))
    return gaps


def image_delimiter_labels(spans: Sequence[Dict[str, Any]]) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    for span in spans:
        kind = str(span.get("kind", ""))
        if not kind.startswith("image_"):
            continue
        name = str(span["name"])
        prefix, modality = name.rsplit("_", 1)
        start = int(span["start"])
        end = int(span["end"])
        labels[start - 1] = f"{prefix}_{modality}_start_of_image"
        labels[end] = f"{prefix}_{modality}_end_of_image"
    return labels


def compact_counts(counter: Counter[str]) -> str:
    return "; ".join(f"{name}:{count}" for name, count in sorted(counter.items()))


def analyze_samples(batch_root: Path, run_names: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Counter[Tuple[str, str]]]:
    sample_rows: List[Dict[str, Any]] = []
    position_rows: List[Dict[str, Any]] = []
    label_counts: Counter[Tuple[str, str]] = Counter()

    for run in run_names:
        run_root = batch_root / "runs" / run
        for sample_dir in sorted(run_root.glob("sample_*")):
            required = ["metadata.json", "spans.json", "token_trace.json"]
            if not all((sample_dir / name).is_file() for name in required):
                continue
            metadata = load_json(sample_dir / "metadata.json")
            spans = load_json(sample_dir / "spans.json")
            token_trace = load_json(sample_dir / "token_trace.json")
            target_step = choose_target_step(metadata, token_trace)
            covered, answer_start, query_pos = covered_intervals(spans, token_trace, target_step)
            gaps = complement_intervals(covered, query_pos + 1)
            labels = image_delimiter_labels(spans)

            counter: Counter[str] = Counter()
            for gap_start, gap_end in gaps:
                for pos in range(gap_start, gap_end):
                    label = labels.get(pos, "UNKNOWN")
                    counter[label] += 1
                    label_counts[(run, label)] += 1
                    position_rows.append({
                        "run": run,
                        "sample_dir": sample_dir.name,
                        "sample_index": metadata.get("sample_index", ""),
                        "sample_id": metadata.get("sample_id", ""),
                        "target_step": target_step,
                        "query_pos": query_pos,
                        "kv_position": pos,
                        "label": label,
                    })

            sample_rows.append({
                "run": run,
                "sample_dir": sample_dir.name,
                "sample_index": metadata.get("sample_index", ""),
                "sample_id": metadata.get("sample_id", ""),
                "target_step": target_step,
                "target_step_source": metadata.get("target_step_source", ""),
                "query_pos": query_pos,
                "answer_start": answer_start,
                "total_kv_positions_seen": query_pos + 1,
                "unnamed_position_count": sum(counter.values()),
                "unknown_position_count": counter.get("UNKNOWN", 0),
                "unnamed_labels": compact_counts(counter),
            })

    return sample_rows, position_rows, label_counts


def summarize_other_attention(batch_root: Path, run_names: Sequence[str]) -> List[Dict[str, Any]]:
    wide_path = batch_root / "attention_distribution_wide.csv"
    if not wide_path.is_file():
        return []
    values: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    with wide_path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            run = row.get("run", "")
            if run not in run_names:
                continue
            value = row.get("other_unnamed", "")
            if value == "":
                continue
            values[(run, int(row["layer"]))].append(float(value))

    rows: List[Dict[str, Any]] = []
    for (run, layer), vals in sorted(values.items()):
        rows.append({
            "run": run,
            "layer": layer,
            "n": len(vals),
            "mean_other_unnamed": statistics.mean(vals),
            "median_other_unnamed": statistics.median(vals),
            "p25_other_unnamed": pctile(vals, 0.25),
            "p75_other_unnamed": pctile(vals, 0.75),
        })
    return rows


def write_markdown(
    path: Path,
    batch_root: Path,
    run_names: Sequence[str],
    sample_rows: Sequence[Dict[str, Any]],
    label_counts: Counter[Tuple[str, str]],
    attention_rows: Sequence[Dict[str, Any]],
) -> None:
    samples_by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        samples_by_run[str(row["run"])].append(row)

    attention_by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in attention_rows:
        attention_by_run[str(row["run"])].append(row)

    lines = [
        "# Unnamed Attention Token Analysis",
        "",
        f"Batch root: `{batch_root}`",
        "",
        "The probe computes `other_unnamed = total - covered_total`, where covered tokens are "
        "`V1 + V2 + bridge_all + text + answer_prefix + current_query`.",
        "",
        "For these runs, unnamed positions are recovered by taking the complement of all named "
        "spans plus the answer-prefix/current-query spans up to the target answer query.",
        "",
        "## Position Summary",
        "",
        "| run | samples | unnamed positions/sample | unknown positions | conclusion |",
        "|:---|---:|---:|---:|:---|",
    ]

    for run in run_names:
        rows = samples_by_run.get(run, [])
        if not rows:
            lines.append(f"| `{run}` | 0 |  |  | missing run output |")
            continue
        unnamed_counts = [int(row["unnamed_position_count"]) for row in rows]
        unknown_total = sum(int(row["unknown_position_count"]) for row in rows)
        unnamed_desc = (
            str(unnamed_counts[0])
            if min(unnamed_counts) == max(unnamed_counts)
            else f"{min(unnamed_counts)}-{max(unnamed_counts)}"
        )
        conclusion = (
            "all unnamed positions are image delimiters"
            if unknown_total == 0
            else "contains non-delimiter unknown positions"
        )
        lines.append(f"| `{run}` | {len(rows)} | {unnamed_desc} | {unknown_total} | {conclusion} |")

    lines.extend([
        "",
        "## Exact Unnamed Token Classes",
        "",
        "Each count below is across all samples in the run. A count of 133 means exactly one such token per BLINK val sample.",
        "",
        "| run | token class | count |",
        "|:---|:---|---:|",
    ])
    for run in run_names:
        for (_, label), count in sorted(
            ((key, count) for key, count in label_counts.items() if key[0] == run),
            key=lambda item: item[0][1],
        ):
            lines.append(f"| `{run}` | `{label}` | {count} |")

    lines.extend([
        "",
        "## Other-Unnamed Attention Mass",
        "",
        "Because raw Q/K tensors were skipped in the batch run, this report cannot split "
        "the attention mass among the 12 delimiter tokens. Since there are zero unknown "
        "positions, the saved `other_unnamed` mass is nevertheless the total attention "
        "to image delimiter/control tokens as a group.",
        "",
        "| run | mean across layers | max layer | max mean | final layer mean |",
        "|:---|---:|---:|---:|---:|",
    ])
    for run in run_names:
        rows = attention_by_run.get(run, [])
        if not rows:
            lines.append(f"| `{run}` |  |  |  |  |")
            continue
        means = [float(row["mean_other_unnamed"]) for row in rows]
        max_row = max(rows, key=lambda row: float(row["mean_other_unnamed"]))
        final_row = max(rows, key=lambda row: int(row["layer"]))
        lines.append(
            f"| `{run}` | {100 * statistics.mean(means):.2f}% | "
            f"{max_row['layer']} | {100 * float(max_row['mean_other_unnamed']):.2f}% | "
            f"{100 * float(final_row['mean_other_unnamed']):.2f}% |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "For BLINK val, `other_unnamed` is not prompt text, answer text, or image patch tokens. "
        "It is exactly the 12 image boundary/control tokens inserted around the VAE and ViT "
        "representations of V1, V2, and the generated bridge.",
        "",
        "Those 12 tokens are:",
        "",
        "- V1 VAE `<image_start>` / `<image_end>`",
        "- V1 ViT `<image_start>` / `<image_end>`",
        "- V2 VAE `<image_start>` / `<image_end>`",
        "- V2 ViT `<image_start>` / `<image_end>`",
        "- bridge VAE `<image_start>` / `<image_end>`",
        "- bridge ViT `<image_start>` / `<image_end>`",
        "",
        "High `other_unnamed` therefore means the answer query is putting substantial mass on "
        "image block boundary/modality markers. It does not mean attention to untracked image patches.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.batch_root / "unnamed_token_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_names:
        run_names = args.run_names
    else:
        run_names = sorted(path.name for path in (args.batch_root / "runs").iterdir() if path.is_dir())

    sample_rows, position_rows, label_counts = analyze_samples(args.batch_root, run_names)
    attention_rows = summarize_other_attention(args.batch_root, run_names)

    write_csv(output_dir / "unnamed_sample_summary.csv", sample_rows)
    write_csv(output_dir / "unnamed_positions.csv", position_rows)
    write_csv(
        output_dir / "unnamed_token_class_counts.csv",
        [
            {"run": run, "token_class": label, "count": count}
            for (run, label), count in sorted(label_counts.items())
        ],
    )
    write_csv(output_dir / "other_unnamed_attention_by_layer.csv", attention_rows)
    write_markdown(
        output_dir / "README.md",
        args.batch_root,
        run_names,
        sample_rows,
        label_counts,
        attention_rows,
    )

    print(f"Wrote {output_dir / 'README.md'}")
    print(f"Wrote {output_dir / 'unnamed_sample_summary.csv'}")
    print(f"Wrote {output_dir / 'unnamed_positions.csv'}")
    print(f"Wrote {output_dir / 'unnamed_token_class_counts.csv'}")
    print(f"Wrote {output_dir / 'other_unnamed_attention_by_layer.csv'}")


if __name__ == "__main__":
    main()
