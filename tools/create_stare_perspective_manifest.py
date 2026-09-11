#!/usr/bin/env python3
"""Create a fixed manifest for STARE perspective attention probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_DATA_FILE = (
    "/path/to/scratch/datasets/STARE/perspective/test-00000-of-00001.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a sample manifest for STARE perspective.",
    )
    parser.add_argument("--data_file", default=DEFAULT_DATA_FILE)
    parser.add_argument(
        "--sample_count",
        default="all",
        help="Number of samples to keep, or 'all' for the whole parquet file.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_sample_count(value: str, dataset_len: int) -> int:
    if str(value).lower() == "all":
        return dataset_len
    count = int(value)
    if count < 0:
        raise ValueError("--sample_count must be non-negative or 'all'")
    if count > dataset_len:
        raise ValueError(f"sample_count={count} exceeds STARE length={dataset_len}")
    return count


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.data_file)
    records = df.to_dict("records")
    sample_count = parse_sample_count(args.sample_count, len(records))

    indexed = []
    for sample_index, record in enumerate(records[:sample_count]):
        indexed.append({
            "sample_index": sample_index,
            "sample_id": str(record.get("qid", f"index_{sample_index}")),
            "sub_task": record.get("category", "stare_perspective"),
            "gt_answer": str(record.get("answer", "")).upper(),
        })

    manifest = {
        "dataset_kind": "stare_perspective",
        "data_file": args.data_file,
        "task": "perspective",
        "split": "test",
        "dataset_len": len(records),
        "sample_count": len(indexed),
        "samples": indexed,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {output} with {len(indexed)} STARE perspective samples")


if __name__ == "__main__":
    main()
