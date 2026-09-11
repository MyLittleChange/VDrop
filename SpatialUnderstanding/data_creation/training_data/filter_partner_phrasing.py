#!/usr/bin/env python3
"""Drop rows with 'partner'-phrased questions from text_thinking jsonls.

These come from upstream perspective_taking generation (~163 of 1500 PT rows),
and the user wants them excluded from training. Reads from text_thinking_v2/
and writes a parallel directory with the same per-category jsonl layout.
"""

import argparse
import glob
import json
import os
from collections import Counter


CATEGORIES = ["counting", "anchor", "relative_distance", "spatial", "perspective_taking"]


def filter_dir(in_dir: str, out_dir: str, dry_run: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    grand_total = 0
    grand_kept = 0
    grand_dropped = 0
    drop_by_cat = Counter()
    for cat in CATEGORIES:
        in_p = os.path.join(in_dir, f"{cat}.jsonl")
        out_p = os.path.join(out_dir, f"{cat}.jsonl")
        if not os.path.exists(in_p):
            print(f"  [skip] {in_p} missing")
            continue
        n = 0
        kept = []
        dropped = 0
        with open(in_p) as f:
            for ln in f:
                r = json.loads(ln)
                n += 1
                q = r["conversations"][0]["value"]
                if "partner" in q.lower():
                    dropped += 1
                    drop_by_cat[cat] += 1
                else:
                    kept.append(ln)
        grand_total += n
        grand_kept += len(kept)
        grand_dropped += dropped
        print(f"  {cat:25s} in={n:5d}  kept={len(kept):5d}  dropped={dropped:4d}")
        if not dry_run:
            with open(out_p, "w") as f:
                f.writelines(kept)
    print()
    print(f"TOTAL  in={grand_total}  kept={grand_kept}  dropped={grand_dropped}")
    if drop_by_cat:
        print(f"  drop by cat: {dict(drop_by_cat)}")
    print(f"  output → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_dir",
        default="/path/to/scratch/infinigen/training_data_mix_all_balance/text_thinking_v2",
    )
    ap.add_argument(
        "--out_dir",
        default="/path/to/scratch/infinigen/training_data_mix_all_balance/text_thinking_v2_no_partner",
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    filter_dir(args.in_dir, args.out_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
