#!/usr/bin/env python3
"""Rewrite mix_all text_thinking JSONL files so every row has a distinct id.

For each id that appears in N>1 rows, the k-th occurrence gets suffix `_a`, `_b`, ...
in row order. Single-occurrence ids stay unchanged. Original files are backed up
to <name>.jsonl.bak (skipped if backup already exists).

Run:
    python rekey_text_thinking_dups.py [--dry_run]
"""

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict


TEXT_THINKING_DIR = "/path/to/scratch/infinigen/training_data_mix_all_balance/text_thinking"
CATEGORIES = ["counting", "anchor", "relative_distance", "spatial", "perspective_taking"]


def suffix_for(k: int) -> str:
    # _a, _b, ..., _z, _aa, _ab, ...
    s = ""
    k_ = k
    while True:
        s = chr(ord("a") + k_ % 26) + s
        k_ = k_ // 26 - 1
        if k_ < 0:
            break
    return "_" + s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_thinking_dir", default=TEXT_THINKING_DIR)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    for cat in CATEGORIES:
        path = os.path.join(args.text_thinking_dir, f"{cat}.jsonl")
        if not os.path.exists(path):
            print(f"  [skip] {path} missing")
            continue

        with open(path) as f:
            rows = [json.loads(ln) for ln in f]

        counts = Counter(r.get("id") for r in rows)
        seen = defaultdict(int)
        n_renamed = 0
        for r in rows:
            sid = r.get("id")
            if counts[sid] > 1:
                k = seen[sid]
                seen[sid] += 1
                r["id"] = f"{sid}{suffix_for(k)}"
                n_renamed += 1

        # Verify uniqueness
        new_counts = Counter(r["id"] for r in rows)
        dup_after = [k for k, v in new_counts.items() if v > 1]
        assert not dup_after, f"{cat}: still has dups after rekey: {dup_after[:5]}"

        print(f"  {cat}: {len(rows)} rows, {n_renamed} ids suffixed, {len(new_counts)} unique ids")

        if args.dry_run:
            continue

        backup = path + ".bak"
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
            print(f"    backed up → {backup}")
        else:
            print(f"    backup already exists at {backup} (skipped)")

        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"    wrote {path}")


if __name__ == "__main__":
    main()
