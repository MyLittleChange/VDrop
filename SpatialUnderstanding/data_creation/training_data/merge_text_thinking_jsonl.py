#!/usr/bin/env python3
"""Concatenate per-category text_thinking jsonls into a single training file
and swap the user prompt from the no-thinking phrasing to the text-thinking
phrasing (matches what create_mix_all_sft_data.py's text_thinking mode emits).

Per-cat rows have:
  user.value = NO_THINKING_PROMPT + "\\n\\n<image><image>\\n" + question

Output rows have:
  user.value = TEXT_THINKING_PROMPT + "\\n\\n<image><image>\\n" + question
  gpt.value  = unchanged ("<think>...</think>\\n<answer>X</answer>")
  image      = unchanged
  id         = unchanged (already source-prefixed)
"""

import argparse
import glob
import json
import os
from collections import Counter


NO_THINKING_PROMPT = (
    "Answer the question directly. "
    "Provide your final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)
TEXT_THINKING_PROMPT = (
    "Think step by step before answering. "
    "Enclose your thinking within <think> </think> tags. "
    "Provide your final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)

CATEGORIES = ["counting", "anchor", "relative_distance", "spatial", "perspective_taking"]


def rewrite_user_value(v: str) -> str:
    """Swap leading NO_THINKING_PROMPT for TEXT_THINKING_PROMPT.

    Falls back to verbatim copy if the prefix is absent (so we never silently
    corrupt an unexpected row format)."""
    if v.startswith(NO_THINKING_PROMPT):
        return TEXT_THINKING_PROMPT + v[len(NO_THINKING_PROMPT):]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_dir",
        default="/path/to/scratch/infinigen/training_data_mix_all_balance/text_thinking_v2_no_partner",
    )
    ap.add_argument(
        "--out_file",
        default="/path/to/scratch/infinigen/training_data_mix_all_balance/text_thinking_v2_no_partner/text_thinking.jsonl",
    )
    args = ap.parse_args()

    n_total = 0
    n_rewritten = 0
    n_passthrough = 0
    cat_counts = Counter()
    src_counts = Counter()

    out_lines = []
    for cat in CATEGORIES:
        p = os.path.join(args.in_dir, f"{cat}.jsonl")
        if not os.path.exists(p):
            print(f"  [skip] {p} missing")
            continue
        with open(p) as f:
            for ln in f:
                r = json.loads(ln)
                n_total += 1
                cat_counts[cat] += 1
                sid = r.get("id", "")
                if sid.startswith("v4_"):
                    src_counts["v4"] += 1
                elif sid.startswith("v5_"):
                    src_counts["v5"] += 1
                elif sid.startswith("ankur_"):
                    src_counts["ankur"] += 1
                else:
                    src_counts["bare"] += 1
                user_v = r["conversations"][0]["value"]
                new_user_v = rewrite_user_value(user_v)
                if new_user_v != user_v:
                    n_rewritten += 1
                else:
                    n_passthrough += 1
                r["conversations"][0]["value"] = new_user_v
                out_lines.append(json.dumps(r))

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"Wrote {args.out_file}")
    print(f"  total rows           : {n_total}")
    print(f"  prompt rewritten     : {n_rewritten}")
    print(f"  prompt passthrough   : {n_passthrough}")
    print(f"  category counts      : {dict(cat_counts)}")
    print(f"  source counts        : {dict(src_counts)}")


if __name__ == "__main__":
    main()
