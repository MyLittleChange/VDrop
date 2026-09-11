#!/usr/bin/env python3
"""Batch runner for answer-token attention probes.

This keeps one BAGEL checkpoint loaded while probing many fixed samples from a
manifest. It writes the same per-sample artifacts as answer_attention_probe.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from answer_attention_probe import (
    build_arg_parser,
    load_probe_inferencer,
    run_probe_with_inferencer,
    save_json,
)


def parse_args() -> argparse.Namespace:
    base_parser = build_arg_parser(add_help=False)
    parser = argparse.ArgumentParser(
        description="Run answer-token attention probes over a fixed sample manifest.",
        parents=[base_parser],
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSON manifest with a samples list containing sample_index entries.",
    )
    parser.add_argument(
        "--run_name",
        required=True,
        help="Name used under output_root/runs/<run_name>/sample_<idx>.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root directory for batch outputs.",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Optional limit for smoke tests; defaults to all manifest samples.",
    )
    parser.add_argument(
        "--shard",
        default=None,
        help="Optional shard spec index/total over the manifest samples, e.g. 0/4.",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Delete existing per-sample output directories before rerunning.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue probing later samples if one sample fails.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Dict[str, Any]:
    manifest = json.loads(path.read_text())
    if "samples" not in manifest or not isinstance(manifest["samples"], list):
        raise ValueError(f"{path} does not contain a list field named 'samples'")
    for item in manifest["samples"]:
        if "sample_index" not in item:
            raise ValueError(f"Manifest sample is missing sample_index: {item}")
    return manifest


def parse_shard(shard: str | None) -> tuple[int, int] | None:
    if not shard:
        return None
    try:
        shard_idx, total_shards = [int(part) for part in shard.split("/", 1)]
    except Exception as exc:
        raise ValueError(f"Invalid --shard {shard!r}; expected index/total") from exc
    if total_shards <= 0 or shard_idx < 0 or shard_idx >= total_shards:
        raise ValueError(f"Invalid --shard {shard!r}; expected 0 <= index < total")
    return shard_idx, total_shards


def selected_samples(
    manifest: Dict[str, Any],
    sample_limit: int | None,
    shard: str | None,
) -> List[Dict[str, Any]]:
    samples = list(manifest["samples"])
    if sample_limit is not None:
        samples = samples[:sample_limit]
    parsed = parse_shard(shard)
    if parsed is not None:
        shard_idx, total_shards = parsed
        total_samples = len(samples)
        shard_size = total_samples // total_shards
        remainder = total_samples % total_shards
        start_idx = shard_idx * shard_size + min(shard_idx, remainder)
        end_idx = start_idx + shard_size + (1 if shard_idx < remainder else 0)
        samples = samples[start_idx:end_idx]
    return samples


def complete_sample(output_dir: Path) -> bool:
    return (
        (output_dir / "metadata.json").is_file()
        and (output_dir / "attention_groups.csv").is_file()
    )


def make_probe_args(args: argparse.Namespace, sample_index: int, output_dir: Path) -> argparse.Namespace:
    probe_args = argparse.Namespace(**vars(args))
    probe_args.sample_id = None
    probe_args.sample_index = int(sample_index)
    probe_args.image1 = None
    probe_args.image2 = None
    probe_args.question = None
    probe_args.options = None
    probe_args.correct_answer_idx = None
    probe_args.output_dir = str(output_dir)
    return probe_args


def run_batch(args: argparse.Namespace) -> Dict[str, Any]:
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    if getattr(args, "dataset_kind", "spatial") == "blink_multiview":
        args.data_dir = manifest.get("data_dir", args.data_dir)
        args.blink_task = manifest.get("task", args.blink_task)
        args.blink_split = manifest.get("split", args.blink_split)
    elif getattr(args, "dataset_kind", "spatial") == "stare_perspective":
        args.stare_data_file = manifest.get("data_file", args.stare_data_file)
    elif args.dataset_json is None:
        args.dataset_json = manifest.get("dataset_json")
    if getattr(args, "dataset_kind", "spatial") == "spatial" and args.dataset_json is None:
        raise ValueError("--dataset_json is required when the manifest does not define dataset_json")

    output_root = Path(args.output_root)
    run_root = output_root / "runs" / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    samples = selected_samples(manifest, args.sample_limit, args.shard)
    shard_label = ""
    parsed_shard = parse_shard(args.shard)
    if parsed_shard is not None:
        shard_label = f"_shard{parsed_shard[0]}of{parsed_shard[1]}"

    status: Dict[str, Any] = {
        "run_name": args.run_name,
        "model_path": args.model_path,
        "manifest": str(manifest_path),
        "dataset_json": args.dataset_json,
        "dataset_kind": getattr(args, "dataset_kind", "spatial"),
        "data_dir": args.data_dir,
        "blink_task": getattr(args, "blink_task", None),
        "blink_split": getattr(args, "blink_split", None),
        "stare_data_file": getattr(args, "stare_data_file", None),
        "shard": args.shard,
        "sample_count": len(samples),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed": [],
        "skipped": [],
        "failed": [],
    }
    status_path = run_root / f"batch_status{shard_label}.json"
    save_json(status_path, status)

    print(f"Loading model once for run {args.run_name}: {args.model_path}", flush=True)
    model, inferencer = load_probe_inferencer(args)

    for ordinal, sample in enumerate(samples, start=1):
        sample_index = int(sample["sample_index"])
        sample_id = sample.get("sample_id", "")
        output_dir = run_root / f"sample_{sample_index}"

        if args.force_rerun and output_dir.exists():
            shutil.rmtree(output_dir)

        if complete_sample(output_dir):
            print(
                f"[{ordinal}/{len(samples)}] SKIP {args.run_name} "
                f"sample_index={sample_index} sample_id={sample_id}",
                flush=True,
            )
            status["skipped"].append({
                "sample_index": sample_index,
                "sample_id": sample_id,
                "output_dir": str(output_dir),
            })
            save_json(status_path, status)
            continue

        print(
            f"[{ordinal}/{len(samples)}] RUN {args.run_name} "
            f"sample_index={sample_index} sample_id={sample_id}",
            flush=True,
        )
        probe_args = make_probe_args(args, sample_index, output_dir)
        started = time.time()
        try:
            metadata = run_probe_with_inferencer(
                probe_args,
                inferencer=inferencer,
                model=model,
                reset_seed=True,
            )
            elapsed_sec = time.time() - started
            status["completed"].append({
                "sample_index": sample_index,
                "sample_id": metadata.get("sample_id", sample_id),
                "predicted_answer": metadata.get("predicted_answer"),
                "target_steps": metadata.get("target_steps", []),
                "elapsed_sec": elapsed_sec,
                "output_dir": str(output_dir),
            })
            print(
                f"[{ordinal}/{len(samples)}] DONE {args.run_name} "
                f"sample_index={sample_index} elapsed_sec={elapsed_sec:.1f}",
                flush=True,
            )
        except Exception as exc:
            elapsed_sec = time.time() - started
            failure = {
                "sample_index": sample_index,
                "sample_id": sample_id,
                "elapsed_sec": elapsed_sec,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "output_dir": str(output_dir),
            }
            status["failed"].append(failure)
            print(
                f"[{ordinal}/{len(samples)}] FAILED {args.run_name} "
                f"sample_index={sample_index}: {exc!r}",
                flush=True,
            )
            if not args.continue_on_error:
                save_json(status_path, status)
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            save_json(status_path, status)

    status["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["completed_count"] = len(status["completed"])
    status["skipped_count"] = len(status["skipped"])
    status["failed_count"] = len(status["failed"])
    save_json(status_path, status)
    print(
        f"Finished {args.run_name}: completed={status['completed_count']} "
        f"skipped={status['skipped_count']} failed={status['failed_count']}",
        flush=True,
    )
    return status


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args()
    status = run_batch(args)
    print(json.dumps({
        "run_name": status["run_name"],
        "completed_count": status.get("completed_count", len(status["completed"])),
        "skipped_count": status.get("skipped_count", len(status["skipped"])),
        "failed_count": status.get("failed_count", len(status["failed"])),
    }, indent=2))


if __name__ == "__main__":
    main()
