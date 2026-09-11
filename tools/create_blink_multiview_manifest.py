#!/usr/bin/env python3
"""Create a fixed manifest for BLINK Multi-view_Reasoning attention probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.run_inference_bagel_blink_multiview import load_blink_multiview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a sample manifest for BLINK Multi-view_Reasoning.",
    )
    parser.add_argument("--data_dir", default="/path/to/scratch/datasets/BLINK")
    parser.add_argument("--task", default="Multi-view_Reasoning")
    parser.add_argument("--split", choices=["val", "test", "val_test"], default="val")
    parser.add_argument("--sample_count", type=int, default=133)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split == "val_test":
        records = []
        for split in ("val", "test"):
            records.extend(load_blink_multiview(args.data_dir, args.task, split))
    else:
        records = load_blink_multiview(args.data_dir, args.task, args.split)
    if args.sample_count > len(records):
        raise SystemExit(
            f"sample_count={args.sample_count} exceeds BLINK split length={len(records)}"
        )

    indexed = []
    for sample_index, record in enumerate(records):
        indexed.append({
            "sample_index": sample_index,
            "sample_id": str(record.get("idx", f"index_{sample_index}")),
            "sub_task": record.get("sub_task", ""),
            "gt_answer": record.get("gt_answer", ""),
        })
    # Keep dataset order from the parquet loader. For val_test this means all
    # val rows first, followed by test rows, which preserves the loader's
    # natural split ordering while still giving a fixed manifest.
    indexed = indexed[:args.sample_count]

    manifest = {
        "dataset_kind": "blink_multiview",
        "data_dir": args.data_dir,
        "task": args.task,
        "split": args.split,
        "dataset_len": len(records),
        "sample_count": len(indexed),
        "samples": indexed,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {output} with {len(indexed)} BLINK samples")


if __name__ == "__main__":
    main()
