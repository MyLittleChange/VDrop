#!/usr/bin/env python3
"""Answer-token attention probe for BAGEL visual thinking.

Runs one BAGEL spatial-reasoning sample, captures FlashAttention Q/K during
final answer decoding, and reports whether the answer query attends to the
generated visual bridge.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.data_utils import pil_img2rgb
from debug_attention import AttentionDebugger, AttentionSpan
from inferencer import GEN_THINK_SYSTEM_PROMPT, InterleaveInferencer
from inference.run_inference_bagel_spatial import (
    THINKING_MODE_PROMPTS,
    extract_answer,
    load_model,
    set_seed,
)


DEFAULT_DATA_DIR = "/path/to/scratch/spatial_collab_dataset"
_BLINK_RECORD_CACHE: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
_STARE_RECORD_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def build_arg_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture answer-token attention to BAGEL visual-thinking bridge images.",
        add_help=add_help,
    )
    parser.add_argument(
        "--model_path",
        default="/path/to/scratch/models/BAGEL-7B-MoT",
        help="Path to BAGEL checkpoint directory.",
    )
    parser.add_argument("--dataset_json", default=None, help="Spatial dataset JSON file.")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--dataset_kind",
        choices=["spatial", "blink_multiview", "stare_perspective"],
        default="spatial",
        help="Dataset loader used by the probe.",
    )
    parser.add_argument("--blink_task", default="Multi-view_Reasoning")
    parser.add_argument("--blink_split", choices=["val", "test", "val_test"], default="val")
    parser.add_argument(
        "--stare_data_file",
        default="/path/to/scratch/datasets/STARE/perspective/test-00000-of-00001.parquet",
        help="Path to STARE perspective parquet file.",
    )
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--image1", default=None, help="Explicit V1 image path.")
    parser.add_argument("--image2", default=None, help="Explicit V2 image path.")
    parser.add_argument("--question", default=None)
    parser.add_argument("--options", nargs="*", default=None)
    parser.add_argument("--correct_answer_idx", type=int, default=None)
    parser.add_argument(
        "--thinking_mode",
        choices=sorted(THINKING_MODE_PROMPTS.keys()),
        default="visual_only_thinking",
    )
    parser.add_argument("--think", dest="think", action="store_true", default=True)
    parser.add_argument("--no_think", dest="think", action="store_false")
    parser.add_argument(
        "--bridge_mode",
        choices=["normal", "blank", "swap", "none"],
        default="normal",
        help="Bridge image fed before final answer decoding.",
    )
    parser.add_argument("--swap_bridge_path", default=None)
    parser.add_argument("--output_dir", default="artifacts/answer_attention_probe")
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--capture_steps", nargs="+", type=int, default=None)
    parser.add_argument(
        "--skip_raw_qk",
        action="store_true",
        help="Do not save raw captured Q/K tensors. Heatmaps and CSV summaries are still written.",
    )
    parser.add_argument(
        "--skip_visualizations",
        action="store_true",
        help="Do not save per-sample bar charts or heatmap overlays.",
    )
    parser.add_argument(
        "--force_bridge",
        action="store_true",
        help="Generate and feed a bridge image even if first text does not emit <image_start>.",
    )
    parser.add_argument("--max_mem_per_gpu", default="80GiB")
    parser.add_argument("--vit_min_size", type=int, default=512)
    parser.add_argument("--max_think_token_n", type=int, default=512)
    parser.add_argument("--max_answer_token_n", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--text_temperature", type=float, default=0.3)
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=2.0)
    parser.add_argument("--cfg_interval_start", type=float, default=0.0)
    parser.add_argument("--cfg_interval_end", type=float, default=1.0)
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--num_timesteps", type=int, default=24)
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument(
        "--cfg_renorm_type",
        choices=["global", "channel", "text_channel"],
        default="text_channel",
    )
    parser.add_argument("--image_shapes", type=int, nargs=2, default=[320, 1024])
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "item"


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def resolve_image_path(path: str, data_dir: str) -> str:
    if path.startswith("images/"):
        return os.path.join(data_dir, path)
    return path.replace(
        "/path/to/scratch",
        "/path/to/scratch",
    )


def load_probe_sample(args: argparse.Namespace) -> Dict[str, Any]:
    if args.image1 and args.image2 and args.question:
        if not args.options:
            raise ValueError("--options is required when using explicit image/question inputs")
        return {
            "sample_id": args.sample_id or "explicit_sample",
            "answerer_image": Image.open(args.image1).convert("RGB"),
            "helper_image": Image.open(args.image2).convert("RGB"),
            "question": args.question,
            "options": args.options,
            "correct_answer_idx": args.correct_answer_idx,
            "correct_answer": (
                args.options[args.correct_answer_idx]
                if args.correct_answer_idx is not None and args.correct_answer_idx < len(args.options)
                else None
            ),
            "question_type": "explicit",
        }

    dataset_kind = getattr(args, "dataset_kind", "spatial")
    if dataset_kind == "blink_multiview":
        return load_blink_multiview_probe_sample(args)
    if dataset_kind == "stare_perspective":
        return load_stare_perspective_probe_sample(args)

    if not args.dataset_json:
        raise ValueError("Provide --dataset_json or explicit --image1/--image2/--question/--options")

    dataset_path = Path(args.dataset_json)
    if not dataset_path.is_absolute():
        dataset_path = Path(args.data_dir) / dataset_path
    with dataset_path.open("r") as f:
        data = json.load(f)
    if not data:
        raise ValueError(f"No samples found in {dataset_path}")

    if args.sample_id is not None:
        matches = [item for item in data if str(item.get("sample_id", "")) == str(args.sample_id)]
        if not matches:
            raise ValueError(f"sample_id={args.sample_id!r} not found in {dataset_path}")
        example = matches[0]
    else:
        example = data[args.sample_index]

    user_1_image_path = resolve_image_path(example["user_1_image_local_path"], args.data_dir)
    user_2_image_path = resolve_image_path(example["user_2_image_local_path"], args.data_dir)
    question = example["question_both_views"].replace(
        "both your and your partner's perspectives",
        "both images",
    )

    if example.get("options_user_2") is not None:
        options = example["options_user_2"]
        correct_answer_idx = example.get("user_2_gt_answer_idx")
        correct_answer = example.get("user_2_gt_answer_text")
    else:
        options = example.get("options_user_1", [])
        correct_answer_idx = example.get("user_1_gt_answer_idx")
        correct_answer = example.get("user_1_gt_answer_text")

    return {
        "sample_id": example.get("sample_id", f"index_{args.sample_index}"),
        "answerer_image": Image.open(user_1_image_path).convert("RGB"),
        "helper_image": Image.open(user_2_image_path).convert("RGB"),
        "question": question,
        "options": options,
        "correct_answer_idx": correct_answer_idx,
        "correct_answer": correct_answer,
        "question_type": example.get("question_type", ""),
        "image_paths": {
            "V1": user_1_image_path,
            "V2": user_2_image_path,
        },
    }


def load_blink_multiview_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    key = (str(args.data_dir), str(args.blink_task), str(args.blink_split))
    if key not in _BLINK_RECORD_CACHE:
        from inference.run_inference_bagel_blink_multiview import load_blink_multiview

        if args.blink_split == "val_test":
            records = []
            for split in ("val", "test"):
                records.extend(load_blink_multiview(args.data_dir, args.blink_task, split))
            _BLINK_RECORD_CACHE[key] = records
        else:
            _BLINK_RECORD_CACHE[key] = load_blink_multiview(
                data_dir=args.data_dir,
                task=args.blink_task,
                split=args.blink_split,
            )
    return _BLINK_RECORD_CACHE[key]


def load_blink_multiview_probe_sample(args: argparse.Namespace) -> Dict[str, Any]:
    from inference.run_inference_bagel_blink_multiview import load_sample

    records = load_blink_multiview_records(args)
    if not records:
        raise ValueError(
            f"No BLINK samples found for {args.data_dir}/{args.blink_task}/{args.blink_split}"
        )

    if args.sample_id is not None:
        matches = [
            (idx, item)
            for idx, item in enumerate(records)
            if str(item.get("idx", "")) == str(args.sample_id)
        ]
        if not matches:
            raise ValueError(f"sample_id={args.sample_id!r} not found in BLINK records")
        sample_index, record = matches[0]
    else:
        sample_index = int(args.sample_index)
        record = records[sample_index]

    loaded = load_sample(record)
    return {
        "sample_id": loaded["sample_id"],
        "answerer_image": loaded["image1"],
        "helper_image": loaded["image2"],
        "question": loaded["question"],
        "options": [],
        "correct_answer_idx": None,
        "correct_answer": loaded["gt_answer"],
        "question_type": loaded.get("sub_task", ""),
        "sub_task": loaded.get("sub_task", ""),
        "dataset_kind": "blink_multiview",
        "sample_index": sample_index,
    }


def load_stare_perspective_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    data_file = str(args.stare_data_file)
    if data_file not in _STARE_RECORD_CACHE:
        import pandas as pd

        df = pd.read_parquet(data_file)
        _STARE_RECORD_CACHE[data_file] = df.to_dict("records")
    return _STARE_RECORD_CACHE[data_file]


def decode_stare_image(img_dict: Dict[str, Any]) -> Image.Image:
    return Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB")


def load_stare_perspective_probe_sample(args: argparse.Namespace) -> Dict[str, Any]:
    from inference.run_inference_bagel_stare_perspective import PERSPECTIVE_TASK_PROMPT

    records = load_stare_perspective_records(args)
    if not records:
        raise ValueError(f"No STARE samples found in {args.stare_data_file}")

    if args.sample_id is not None:
        matches = [
            (idx, item)
            for idx, item in enumerate(records)
            if str(item.get("qid", "")) == str(args.sample_id)
        ]
        if not matches:
            raise ValueError(f"sample_id={args.sample_id!r} not found in STARE records")
        sample_index, record = matches[0]
    else:
        sample_index = int(args.sample_index)
        record = records[sample_index]

    images = record.get("images")
    if images is None:
        images = []
    if len(images) < 2:
        raise ValueError(
            f"STARE sample {record.get('qid', sample_index)!r} has {len(images)} images; expected 2"
        )

    return {
        "sample_id": str(record.get("qid", f"index_{sample_index}")),
        "answerer_image": decode_stare_image(images[0]),
        "helper_image": decode_stare_image(images[1]),
        "question": PERSPECTIVE_TASK_PROMPT,
        "options": [],
        "correct_answer_idx": None,
        "correct_answer": str(record.get("answer", "")).upper(),
        "question_type": record.get("category", "stare_perspective"),
        "sub_task": record.get("category", "stare_perspective"),
        "dataset_kind": "stare_perspective",
        "sample_index": sample_index,
        "other_info": record.get("other_info", ""),
    }


def make_full_prompt(sample: Dict[str, Any], thinking_mode: str) -> str:
    if not sample.get("options"):
        return THINKING_MODE_PROMPTS[thinking_mode] + "\n\nQUESTION: " + sample["question"]

    options_str = "\n".join(
        f"{chr(65 + i)}) {option}" for i, option in enumerate(sample["options"])
    )
    full_question = f"\nQUESTION:{sample['question']}\n\nOPTIONS:{options_str}"
    return THINKING_MODE_PROMPTS[thinking_mode] + "\n" + full_question


def prepare_context_image(inferencer: InterleaveInferencer, image: Image.Image) -> Image.Image:
    return inferencer.vae_transform.resize_transform(pil_img2rgb(image))


def image_grid_sizes(inferencer: InterleaveInferencer, image: Image.Image) -> Dict[str, Tuple[int, int]]:
    model = inferencer.model
    vae_image = inferencer.vae_transform.resize_transform(image)
    vit_image = inferencer.vit_transform.resize_transform(image)
    return {
        "vae": (
            vae_image.size[1] // model.latent_downsample,
            vae_image.size[0] // model.latent_downsample,
        ),
        "vit": (
            vit_image.size[1] // model.vit_patch_size,
            vit_image.size[0] // model.vit_patch_size,
        ),
    }


def add_text_with_span(
    inferencer: InterleaveInferencer,
    gen_context: Dict[str, Any],
    text: str,
    spans: List[AttentionSpan],
    name: str,
) -> Dict[str, Any]:
    start = int(gen_context["kv_lens"][0])
    gen_context = inferencer.update_context_text(text, gen_context)
    end = int(gen_context["kv_lens"][0])
    spans.append(AttentionSpan(name=name, start=start, end=end, kind="text"))
    return gen_context


def add_image_with_spans(
    inferencer: InterleaveInferencer,
    gen_context: Dict[str, Any],
    image: Image.Image,
    spans: List[AttentionSpan],
    prefix: str,
    source: str,
    add_vae: bool = True,
    add_vit: bool = True,
) -> Tuple[Dict[str, Any], Image.Image]:
    prepared = prepare_context_image(inferencer, image)
    grids = image_grid_sizes(inferencer, prepared)

    if add_vae:
        start = int(gen_context["kv_lens"][0])
        gen_context = inferencer.update_context_image(prepared, gen_context, vae=True, vit=False)
        end = int(gen_context["kv_lens"][0])
        spans.append(
            AttentionSpan(
                name=f"{prefix}_vae",
                start=start + 1,
                end=end - 1,
                kind="image_vae",
                grid_size=grids["vae"],
                source=source,
            )
        )

    if add_vit:
        start = int(gen_context["kv_lens"][0])
        gen_context = inferencer.update_context_image(prepared, gen_context, vae=False, vit=True)
        end = int(gen_context["kv_lens"][0])
        spans.append(
            AttentionSpan(
                name=f"{prefix}_vit",
                start=start + 1,
                end=end - 1,
                kind="image_vit",
                grid_size=grids["vit"],
                source=source,
            )
        )

    return gen_context, prepared


def annotate_token_trace(token_trace: List[Dict[str, Any]]) -> str:
    cumulative = ""
    for item in token_trace:
        text = item.get("predicted_text", "")
        item["predicted_char_start"] = len(cumulative)
        cumulative += text
        item["predicted_char_end"] = len(cumulative)
        item["cumulative_predicted_text"] = cumulative
    return cumulative


def find_answer_prediction_steps(token_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    text = annotate_token_trace(token_trace)
    patterns = [
        ("answer_tag", re.compile(r"<answer>\s*\(?([A-D])", re.IGNORECASE)),
        ("final_answer", re.compile(r"Final\s+Answer\s*:?\s*\(?([A-D])", re.IGNORECASE)),
    ]
    for label, pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        char_end = match.end(1)
        for item in token_trace:
            if item["predicted_char_start"] < char_end <= item["predicted_char_end"]:
                return [{
                    "step": int(item["step"]),
                    "answer": match.group(1).upper(),
                    "pattern": label,
                    "predicted_text": item.get("predicted_text", ""),
                }]

    for item in token_trace:
        token_text = item.get("predicted_text", "")
        match = re.search(r"\b([A-D])\b", token_text)
        if match:
            return [{
                "step": int(item["step"]),
                "answer": match.group(1).upper(),
                "pattern": "token_fallback",
                "predicted_text": token_text,
            }]
    return []


def attention_groups(spans: Sequence[AttentionSpan]) -> Dict[str, List[str]]:
    text_names = [span.name for span in spans if span.kind == "text"]
    groups = {
        "V1": [span.name for span in spans if span.name.startswith("V1_")],
        "V2": [span.name for span in spans if span.name.startswith("V2_")],
        "bridge_vae": [span.name for span in spans if span.name == "bridge_vae"],
        "bridge_vit": [span.name for span in spans if span.name == "bridge_vit"],
        "bridge_all": [span.name for span in spans if span.name.startswith("bridge_")],
        "text": text_names,
    }
    return {name: values for name, values in groups.items() if values}


def build_answer_token_ledger(token_trace: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ledger: List[Dict[str, Any]] = []
    for item in token_trace:
        positions = item.get("query_kv_positions") or []
        if not positions:
            continue
        pos = int(positions[0])
        ledger.append({
            "step": int(item["step"]),
            "query_span": {
                "name": f"answer_query_step_{item['step']}",
                "start": pos,
                "end": pos + 1,
                "kind": "answer_query_token",
            },
            "query_text": item.get("query_text", ""),
            "predicted_text": item.get("predicted_text", ""),
            "predicted_token_ids": item.get("predicted_token_ids", []),
        })
    return ledger


def summarize_attention(
    debugger: AttentionDebugger,
    spans: Sequence[AttentionSpan],
    token_trace: Sequence[Dict[str, Any]],
    target_steps: Sequence[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    span_by_name = {span.name: span for span in spans}
    trace_by_step = {int(item["step"]): item for item in token_trace}
    answer_start = None
    if token_trace and token_trace[0].get("query_kv_positions"):
        answer_start = int(token_trace[0]["query_kv_positions"][0])
    rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    groups = attention_groups(spans)

    for step in target_steps:
        if step not in debugger.captured_data:
            continue
        for layer in sorted(debugger.captured_data[step]):
            masses = debugger.summarize_query_attention_to_spans(
                step_idx=step,
                layer_idx=layer,
                spans=list(spans),
                query_idx=0,
            )
            token_item = trace_by_step.get(step, {})
            query_positions = token_item.get("query_kv_positions") or []
            query_pos = int(query_positions[0]) if query_positions else None
            for name, mass in masses.items():
                rows.append({
                    "step": step,
                    "layer": layer,
                    "span": name,
                    "kind": span_by_name[name].kind,
                    "attention_mass": mass,
                    "predicted_text": token_item.get("predicted_text", ""),
                    "query_kv_position": (token_item.get("query_kv_positions") or [None])[0],
                })

            total_mass = float(debugger.compute_attn_weights(
                step_idx=step,
                layer_idx=layer,
                seq_idx=0,
            )[:, 0, :].sum(dim=-1).mean().item())

            named_total = float(
                sum(masses.get(name, 0.0) for name in groups.get("V1", []))
                + sum(masses.get(name, 0.0) for name in groups.get("V2", []))
                + sum(masses.get(name, 0.0) for name in groups.get("bridge_all", []))
                + sum(masses.get(name, 0.0) for name in groups.get("text", []))
            )
            answer_prefix_mass = 0.0
            current_query_mass = 0.0
            if answer_start is not None and query_pos is not None:
                if query_pos > answer_start:
                    answer_prefix = AttentionSpan(
                        name="answer_prefix",
                        start=answer_start,
                        end=query_pos,
                        kind="answer_prefix",
                    )
                    answer_prefix_mass = float(debugger.extract_query_attention_to_span(
                        step_idx=step,
                        layer_idx=layer,
                        span=answer_prefix,
                        query_idx=0,
                    ).item())
                current_query = AttentionSpan(
                    name="current_query",
                    start=query_pos,
                    end=query_pos + 1,
                    kind="current_query",
                )
                current_query_mass = float(debugger.extract_query_attention_to_span(
                    step_idx=step,
                    layer_idx=layer,
                    span=current_query,
                    query_idx=0,
                ).item())

            covered_total = named_total + answer_prefix_mass + current_query_mass
            other_mass = max(0.0, total_mass - covered_total)
            for group_name, span_names in groups.items():
                group_rows.append({
                    "step": step,
                    "layer": layer,
                    "group": group_name,
                    "attention_mass": float(sum(masses.get(name, 0.0) for name in span_names)),
                    "predicted_text": token_item.get("predicted_text", ""),
                })
            for group_name, mass in [
                ("answer_prefix", answer_prefix_mass),
                ("current_query", current_query_mass),
                ("named_total", named_total),
                ("covered_total", covered_total),
                ("other_unnamed", other_mass),
                ("total", total_mass),
            ]:
                group_rows.append({
                    "step": step,
                    "layer": layer,
                    "group": group_name,
                    "attention_mass": mass,
                    "predicted_text": token_item.get("predicted_text", ""),
                })

    return rows, group_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_heatmap(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    values = values - values.min()
    max_value = values.max()
    if max_value > 0:
        values = values / max_value
    return values


def save_heatmap_overlay(
    path: Path,
    image: Image.Image,
    heatmap: np.ndarray,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    heatmap = normalize_heatmap(heatmap)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(image)
    ax.imshow(
        heatmap,
        cmap="inferno",
        alpha=0.55,
        extent=(0, image.size[0], image.size[1], 0),
        interpolation="bilinear",
    )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_group_bar_chart(path: Path, rows: Sequence[Dict[str, Any]], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["group"] for row in rows]
    values = [row["attention_mass"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_ylabel("attention mass")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_visualizations(
    debugger: AttentionDebugger,
    spans: Sequence[AttentionSpan],
    group_rows: Sequence[Dict[str, Any]],
    token_trace: Sequence[Dict[str, Any]],
    target_steps: Sequence[int],
    images_by_source: Dict[str, Image.Image],
    output_dir: Path,
) -> None:
    trace_by_step = {int(item["step"]): item for item in token_trace}
    image_spans = [span for span in spans if span.grid_size is not None]

    for step in target_steps:
        if step not in debugger.captured_data:
            continue
        token_text = trace_by_step.get(step, {}).get("predicted_text", "")
        for layer in sorted(debugger.captured_data[step]):
            grouped = [
                row for row in group_rows
                if row["step"] == step and row["layer"] == layer
                and row["group"] in {
                    "V1",
                    "V2",
                    "bridge_vae",
                    "bridge_vit",
                    "text",
                    "answer_prefix",
                    "current_query",
                    "other_unnamed",
                }
            ]
            save_group_bar_chart(
                output_dir / f"attention_groups_step{step}_layer{layer}.png",
                grouped,
                f"Prediction step {step}: {token_text!r}, layer {layer}",
            )

            for span in image_spans:
                if span.source not in images_by_source:
                    continue
                heatmap = debugger.extract_query_to_grid_heatmap(
                    step_idx=step,
                    layer_idx=layer,
                    span=span,
                    query_idx=0,
                ).numpy()
                filename = f"overlay_step{step}_layer{layer}_{sanitize_filename(span.name)}.png"
                save_heatmap_overlay(
                    output_dir / filename,
                    images_by_source[span.source],
                    heatmap,
                    f"{span.name} attention, step {step}, layer {layer}, token {token_text!r}",
                )


def load_probe_inferencer(args: argparse.Namespace) -> Tuple[Any, InterleaveInferencer]:
    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(
        args.model_path,
        args.max_mem_per_gpu,
        args.vit_min_size,
    )
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    return model, inferencer


def run_probe_with_inferencer(
    args: argparse.Namespace,
    inferencer: InterleaveInferencer,
    model: Any,
    reset_seed: bool = True,
) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if reset_seed:
        set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this process. Run the probe from a GPU-visible "
            "interactive job or launch Jupyter inside the same GPU allocation."
        )

    sample = load_probe_sample(args)
    full_prompt = make_full_prompt(sample, args.thinking_mode)

    inference_hyper = {
        "do_sample": args.do_sample,
        "temperature": args.text_temperature,
    }
    image_hyper = {
        "cfg_text_scale": args.cfg_text_scale,
        "cfg_img_scale": args.cfg_img_scale,
        "cfg_interval": (args.cfg_interval_start, args.cfg_interval_end),
        "timestep_shift": args.timestep_shift,
        "num_timesteps": args.num_timesteps,
        "cfg_renorm_min": args.cfg_renorm_min,
        "cfg_renorm_type": args.cfg_renorm_type,
    }

    spans: List[AttentionSpan] = []
    output_texts: List[str] = []
    images_by_source: Dict[str, Image.Image] = {}
    gen_context = inferencer.init_gen_context()
    cfg_text_context = deepcopy(gen_context)
    cfg_img_context = deepcopy(gen_context)

    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
        if args.think:
            gen_context = add_text_with_span(
                inferencer,
                gen_context,
                GEN_THINK_SYSTEM_PROMPT,
                spans,
                "generic_think_system",
            )
            cfg_img_context = inferencer.update_context_text(GEN_THINK_SYSTEM_PROMPT, cfg_img_context)

        gen_context, v1_prepared = add_image_with_spans(
            inferencer,
            gen_context,
            sample["answerer_image"],
            spans,
            prefix="V1",
            source="V1",
        )
        images_by_source["V1"] = v1_prepared
        cfg_text_context = deepcopy(gen_context)

        gen_context, v2_prepared = add_image_with_spans(
            inferencer,
            gen_context,
            sample["helper_image"],
            spans,
            prefix="V2",
            source="V2",
        )
        images_by_source["V2"] = v2_prepared
        cfg_text_context = deepcopy(gen_context)

        cfg_text_context = deepcopy(gen_context)
        gen_context = add_text_with_span(
            inferencer,
            gen_context,
            full_prompt,
            spans,
            "question_prompt",
        )
        cfg_img_context = inferencer.update_context_text(full_prompt, cfg_img_context)

        capture_first_text = args.bridge_mode == "none"
        debugger = AttentionDebugger(
            model=model,
            layers_to_monitor=args.layers,
            capture_timesteps=args.capture_steps,
        )

        if capture_first_text:
            debugger.enable()
        try:
            first_text, first_trace = inferencer.gen_text(
                gen_context,
                max_length=args.max_think_token_n,
                debug_attn_callback=debugger.text_step_callback if capture_first_text else None,
                return_token_trace=True,
                **inference_hyper,
            )
        finally:
            if capture_first_text:
                debugger.disable()
        output_texts.append(first_text)

        final_text = first_text
        final_trace = first_trace
        generated_bridge = None
        bridge_used = None

        should_generate_bridge = (
            args.bridge_mode != "none"
            and ("<image_start>" in first_text or args.force_bridge)
        )
        # Visual-thinking inference is a two-stage decode. First, BAGEL writes a
        # visual plan, usually ending with <image_start>. If bridge mode is
        # enabled, that generated text must be appended to the KV context before
        # gen_image so the bridge image is conditioned on the plan. The selected
        # bridge image is then appended as VAE/ViT spans, and only the following
        # answer-text decode is captured for answer-token attention.
        if should_generate_bridge:
            gen_context = add_text_with_span(
                inferencer,
                gen_context,
                first_text,
                spans,
                "visual_plan_text",
            )
            generated_bridge = inferencer.gen_image(
                tuple(args.image_shapes),
                gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                **image_hyper,
            )
            generated_bridge.save(output_dir / "generated_bridge.png")

            if args.bridge_mode == "blank":
                bridge_used = Image.new("RGB", generated_bridge.size, (127, 127, 127))
            elif args.bridge_mode == "swap":
                if not args.swap_bridge_path:
                    raise ValueError("--swap_bridge_path is required for --bridge_mode swap")
                bridge_used = Image.open(args.swap_bridge_path).convert("RGB")
            else:
                bridge_used = generated_bridge
            bridge_used.save(output_dir / "bridge_used_for_answer.png")

            gen_context, bridge_prepared = add_image_with_spans(
                inferencer,
                gen_context,
                bridge_used,
                spans,
                prefix="bridge",
                source="bridge",
            )
            images_by_source["bridge"] = bridge_prepared

            debugger.clear()
            debugger.enable()
            try:
                final_text, final_trace = inferencer.gen_text(
                    gen_context,
                    max_length=args.max_answer_token_n,
                    debug_attn_callback=debugger.text_step_callback,
                    return_token_trace=True,
                    **inference_hyper,
                )
            finally:
                debugger.disable()
            output_texts.append(final_text)
        elif args.bridge_mode == "none" and "<image_start>" in first_text:
            gen_context = add_text_with_span(
                inferencer,
                gen_context,
                first_text,
                spans,
                "visual_plan_text",
            )
            debugger.clear()
            debugger.enable()
            try:
                final_text, final_trace = inferencer.gen_text(
                    gen_context,
                    max_length=args.max_answer_token_n,
                    debug_attn_callback=debugger.text_step_callback,
                    return_token_trace=True,
                    **inference_hyper,
                )
            finally:
                debugger.disable()
            output_texts.append(final_text)
        elif not capture_first_text:
            # The model answered before generating an image. Capture a second run
            # from the same context so the user still gets answer-token attention.
            debugger.clear()
            debugger.enable()
            try:
                final_text, final_trace = inferencer.gen_text(
                    gen_context,
                    max_length=args.max_answer_token_n,
                    debug_attn_callback=debugger.text_step_callback,
                    return_token_trace=True,
                    **inference_hyper,
                )
            finally:
                debugger.disable()
            output_texts[-1] = final_text

    final_answer = extract_answer(final_text)
    answer_steps = find_answer_prediction_steps(final_trace)
    if answer_steps:
        target_steps = [item["step"] for item in answer_steps]
        target_step_source = "answer_token"
    elif final_trace:
        target_steps = [int(final_trace[-1]["step"])]
        target_step_source = "last_generated_token_fallback"
    else:
        target_steps = []
        target_step_source = "none"

    span_rows, group_rows = summarize_attention(
        debugger=debugger,
        spans=spans,
        token_trace=final_trace,
        target_steps=target_steps,
    )

    (output_dir / "final_text.txt").write_text(final_text)
    (output_dir / "all_text_outputs.txt").write_text("\n\n".join(output_texts))
    sample["answerer_image"].save(output_dir / "input_V1.png")
    sample["helper_image"].save(output_dir / "input_V2.png")
    save_json(output_dir / "spans.json", [span.to_dict() for span in spans])
    save_json(output_dir / "answer_token_spans.json", build_answer_token_ledger(final_trace))
    save_json(output_dir / "token_trace.json", final_trace)
    save_json(output_dir / "attention_summary.json", {
        "span_rows": span_rows,
        "group_rows": group_rows,
    })
    write_csv(output_dir / "attention_spans.csv", span_rows)
    write_csv(output_dir / "attention_groups.csv", group_rows)
    if not args.skip_raw_qk:
        debugger.save_raw_data(str(output_dir / "raw_qk"))
    if not args.skip_visualizations:
        save_visualizations(
            debugger=debugger,
            spans=spans,
            group_rows=group_rows,
            token_trace=final_trace,
            target_steps=target_steps,
            images_by_source=images_by_source,
            output_dir=output_dir,
        )

    metadata = {
        "model_path": args.model_path,
        "model_name": Path(args.model_path).name,
        "sample_id": sample["sample_id"],
        "dataset_kind": sample.get("dataset_kind", getattr(args, "dataset_kind", "spatial")),
        "sub_task": sample.get("sub_task"),
        "sample_index": sample.get("sample_index", args.sample_index),
        "question_type": sample.get("question_type"),
        "question": sample["question"],
        "options": sample["options"],
        "correct_answer_idx": sample.get("correct_answer_idx"),
        "correct_answer": sample.get("correct_answer"),
        "predicted_answer": final_answer,
        "thinking_mode": args.thinking_mode,
        "think": args.think,
        "bridge_mode": args.bridge_mode,
        "answer_steps": answer_steps,
        "target_steps": target_steps,
        "target_step_source": target_step_source,
        "layers": sorted(debugger.layers_to_monitor),
        "skip_raw_qk": args.skip_raw_qk,
        "skip_visualizations": args.skip_visualizations,
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "metadata.json", metadata)
    return metadata


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this process. Run the probe from a GPU-visible "
            "interactive job or launch Jupyter inside the same GPU allocation."
        )
    model, inferencer = load_probe_inferencer(args)
    return run_probe_with_inferencer(args, inferencer=inferencer, model=model, reset_seed=False)


def main() -> None:
    args = parse_args()
    metadata = run_probe(args)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
