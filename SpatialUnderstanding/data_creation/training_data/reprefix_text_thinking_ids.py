#!/usr/bin/env python3
"""Re-prefix sample_ids in mix_all training jsonls with their source.

For each row, infer the source ('v4'/'v5'/'ankur') from `row["image"][0]` and
rewrite `row["id"]` from `<base>` to `<source>_<base>`. Idempotent: rows already
prefixed are skipped. Original files are backed up to `<file>.preprefix.bak`
(skipped if backup already exists).

Modes:
  --mode text_thinking : rewrite the 5 category jsonls under text_thinking/
  --mode no_thinking   : rewrite no_thinking/no_thinking.jsonl
  --mode both          : do both (default)

Run:
    python reprefix_text_thinking_ids.py [--mode {text_thinking,no_thinking,both}] [--dry_run]
"""

import argparse
import json
import os
import shutil
from collections import Counter


BALANCE_DIR = (
    "/path/to/scratch/infinigen/training_data_mix_all_balance"
)
TEXT_THINKING_DIR = os.path.join(BALANCE_DIR, "text_thinking")
NO_THINKING_FILE = os.path.join(BALANCE_DIR, "no_thinking", "no_thinking.jsonl")
CATEGORIES = ["counting", "anchor", "relative_distance", "spatial", "perspective_taking"]
SOURCE_PREFIXES = ("v4_", "v5_", "ankur_")


def infer_source_from_path(image_path: str) -> str:
    """Mirrors create_mix_all_sft_data.infer_source_from_path /
    annotate_text_reasoning.parse_source_from_image_path."""
    if not image_path:
        return "unknown"
    if "infinigen/spatial/" in image_path:
        return "v5"
    if "outputs_rendered/" in image_path:
        return "v4"
    if "spatial_collab_dataset/scenes/" in image_path:
        return "ankur"
    return "unknown"


def reprefix_file(path: str, label: str, dry_run: bool) -> None:
    if not os.path.exists(path):
        print(f"  [skip] {path} missing")
        return

    with open(path) as f:
        rows = [json.loads(ln) for ln in f]

    n_rewritten = 0
    n_already = 0
    n_unknown = 0
    src_dist = Counter()
    for r in rows:
        sid = r.get("id", "")
        if not sid:
            continue
        if sid.startswith(SOURCE_PREFIXES):
            n_already += 1
            continue
        img_list = r.get("image", [])
        img0 = img_list[0] if img_list else ""
        src = infer_source_from_path(img0)
        src_dist[src] += 1
        if src == "unknown":
            n_unknown += 1
            continue
        r["id"] = f"{src}_{sid}"
        n_rewritten += 1

    print(
        f"  {label}: {len(rows)} rows | rewrote {n_rewritten} | already prefixed {n_already} "
        f"| unknown source {n_unknown} | src_dist={dict(src_dist)}"
    )

    if dry_run:
        return

    backup = path + ".preprefix.bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"    backed up → {backup}")
    else:
        print(f"    backup already exists at {backup} (skipped)")

    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"    wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_thinking_dir", default=TEXT_THINKING_DIR)
    parser.add_argument("--no_thinking_file", default=NO_THINKING_FILE)
    parser.add_argument(
        "--mode",
        choices=["text_thinking", "no_thinking", "both"],
        default="both",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.mode in ("text_thinking", "both"):
        print("=== text_thinking ===")
        for cat in CATEGORIES:
            path = os.path.join(args.text_thinking_dir, f"{cat}.jsonl")
            reprefix_file(path, cat, args.dry_run)

    if args.mode in ("no_thinking", "both"):
        print("=== no_thinking ===")
        reprefix_file(args.no_thinking_file, "no_thinking", args.dry_run)


if __name__ == "__main__":
    main()
