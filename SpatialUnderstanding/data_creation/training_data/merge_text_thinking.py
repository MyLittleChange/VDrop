#!/usr/bin/env python3
"""Merge per-category text-thinking jsonls into a single training-ready jsonl.

Reads
    /path/to/scratch/infinigen/training_data_mix_all_balance/text_thinking/
        counting.jsonl
        anchor.jsonl
        relative_distance.jsonl
        spatial.jsonl
        perspective_taking.jsonl
and writes
    /path/to/scratch/infinigen/training_data_mix_all_balance/text_thinking/text_thinking.jsonl

A `manifest.json` summarising counts per category is also written.
"""

import argparse
import json
import os
from collections import Counter

DEFAULT_DIR = "/path/to/scratch/infinigen/training_data_mix_all_balance/text_thinking"
CATEGORIES = ["counting", "anchor", "relative_distance", "spatial", "perspective_taking"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--out", default=None,
                    help="Default: <dir>/text_thinking.jsonl")
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.dir, "text_thinking.jsonl")
    counts = Counter()
    failed = Counter()
    total = 0
    with open(out_path, "w") as out:
        for cat in CATEGORIES:
            path = os.path.join(args.dir, f"{cat}.jsonl")
            if not os.path.exists(path):
                print(f"[skip] {path} missing")
                continue
            with open(path) as f:
                for ln in f:
                    out.write(ln)
                    counts[cat] += 1
                    total += 1
            fail_path = os.path.join(args.dir, f"{cat}_failed.jsonl")
            if os.path.exists(fail_path):
                with open(fail_path) as f:
                    for _ in f:
                        failed[cat] += 1

    manifest = {
        "output": out_path,
        "total_rows": total,
        "rows_per_category": dict(counts),
        "failed_per_category": dict(failed),
    }
    manifest_path = os.path.join(args.dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {total} rows to {out_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
