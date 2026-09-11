#!/usr/bin/env python3
"""Build interleaved-mode parquet by joining existing visual_only parquet
with annotated text_thinking JSONL.

Why:
  Re-running create_mix_all_sft_data.py --thinking_mode interleaved drops ~24%
  of samples because the upstream sample pool has drifted (V4/V5/ankur
  sample_id collisions + bad_panoramas churn + missing ankur loader). The
  visual_only/*.parquet (7912 rows) and text_thinking/*.jsonl (7828 rows)
  files we already have on disk ARE the canonical pool. Join them by
  question text (with cam0-bytes md5 tie-breaker) and emit interleaved
  rows directly.

Output schema matches visual_only:
  - image_list:        list<binary>   [cam0, cam1, panorama]   (passed through)
  - instruction_list:  list<string>   [visual-thinking prompt] (passed through)
  - output_text_list:  list<string>   ["<think>{think}</think><image_start>",
                                       "<image_end><answer>{X}</answer>"]
"""

import argparse
import glob
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


VO_PREFIX = (
    "Think visually to answer the question. "
    "Enclose your visual thinking within <image_start> </image_end>.\n"
)
TT_PREFIX = (
    "Answer the question directly. Provide your final answer wrapped in "
    "<answer></answer> tags, i.e.<answer> answer here </answer>.\n\n<image><image>\n"
)

CATEGORIES = ["counting", "anchor", "relative_distance", "spatial", "perspective_taking"]


def remap_path(p: str) -> str:
    if not p:
        return p
    p = p.replace("/path/to/scratch", "/path/to/scratch")
    p = p.replace("/path/to/scratch", "/path/to/scratch")
    if not p.startswith("/"):
        p = "/network/scratch/" + p
    return p


def norm_vo(q: str) -> str:
    return q[len(VO_PREFIX):].strip() if q.startswith(VO_PREFIX) else q.strip()


def norm_tt(q: str) -> str:
    return q[len(TT_PREFIX):].strip() if q.startswith(TT_PREFIX) else q.strip()


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse_think_answer(gpt_value: str):
    m_think = THINK_RE.search(gpt_value)
    m_ans = ANSWER_RE.search(gpt_value)
    if not m_think or not m_ans:
        return None
    return m_think.group(1).strip(), m_ans.group(1).strip()


def file_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def png_bytes_md5_normalized(b: bytes) -> str:
    """Hash decoded raw RGB pixel bytes so PNG re-encoding doesn't change the key."""
    im = Image.open(io.BytesIO(b)).convert("RGB")
    return hashlib.md5(im.tobytes()).hexdigest()


def file_md5_normalized(path: str) -> str:
    im = Image.open(path).convert("RGB")
    return hashlib.md5(im.tobytes()).hexdigest()


def load_text_thinking(text_thinking_dirs, drop_partner_phrasing=True):
    """
    Build a question-keyed lookup. Returns:
      qkey -> list of {think, answer, image0_path, id, cat}
    Multiple entries per qkey only when ≥2 text_thinking rows share a question.

    drop_partner_phrasing: if True, skip rows whose question contains "partner"
    (perspective_taking phrasings like "your partner's perspective"). These are
    upstream artifacts; the user wants them excluded from the interleaved set.
    """
    by_q = defaultdict(list)
    n_rows = 0
    n_parse_fail = 0
    n_partner_skipped = 0
    partner_by_cat = Counter()
    for tdir in text_thinking_dirs:
        for cat in CATEGORIES:
            p = os.path.join(tdir, f"{cat}.jsonl")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                for ln in f:
                    r = json.loads(ln)
                    n_rows += 1
                    q = norm_tt(r["conversations"][0]["value"])
                    if drop_partner_phrasing and "partner" in q.lower():
                        n_partner_skipped += 1
                        partner_by_cat[cat] += 1
                        continue
                    parsed = parse_think_answer(r["conversations"][1]["value"])
                    if parsed is None:
                        n_parse_fail += 1
                        continue
                    think, answer = parsed
                    by_q[q].append({
                        "think": think,
                        "answer": answer,
                        "image0_path": r["image"][0],
                        "id": r["id"],
                        "cat": cat,
                    })
    print(f"[text_thinking] loaded {n_rows} rows from {text_thinking_dirs}")
    print(f"[text_thinking]   parse failures: {n_parse_fail}")
    if drop_partner_phrasing:
        print(f"[text_thinking]   dropped 'partner'-phrased rows: {n_partner_skipped} {dict(partner_by_cat)}")
    print(f"[text_thinking]   unique questions: {len(by_q)}")
    multi = sum(1 for v in by_q.values() if len(v) > 1)
    print(f"[text_thinking]   questions with ≥2 candidates: {multi}")
    return by_q


def write_chunk(out_dir, chunk_idx, image_lists, instruction_lists, output_text_lists):
    table = pa.table({
        "image_list": pa.array(image_lists, type=pa.list_(pa.binary())),
        "instruction_list": pa.array(instruction_lists, type=pa.list_(pa.string())),
        "output_text_list": pa.array(output_text_lists, type=pa.list_(pa.string())),
    })
    out = os.path.join(out_dir, f"chunk_{chunk_idx}.parquet")
    pq.write_table(table, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--visual_only_dir",
        default="/path/to/scratch/infinigen/training_data_mix_all_balance/visual_only",
    )
    ap.add_argument(
        "--text_thinking_dirs",
        nargs="+",
        default=[
            "/path/to/scratch/infinigen/training_data_mix_all_balance/text_thinking",
        ],
    )
    ap.add_argument(
        "--out_dir",
        default="/path/to/scratch/infinigen/training_data_mix_all_balance/interleaved",
    )
    ap.add_argument(
        "--unmatched_jsonl",
        default="/path/to/scratch/infinigen/training_data_mix_all_balance/unmatched_visual_only.jsonl",
    )
    ap.add_argument("--rows_per_chunk", type=int, default=100)
    ap.add_argument(
        "--match_mode",
        choices=["raw_bytes", "decoded_pixels"],
        default="raw_bytes",
        help="raw_bytes: md5 of PNG bytes (fast, only matches if encodings agree). "
             "decoded_pixels: md5 of decoded RGB pixels (slower, robust to re-encoding).",
    )
    ap.add_argument(
        "--keep_partner_phrasing",
        action="store_true",
        help="Keep perspective_taking rows whose question contains 'partner' "
             "(default: drop them).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    by_q = load_text_thinking(
        args.text_thinking_dirs,
        drop_partner_phrasing=not args.keep_partner_phrasing,
    )

    # Tie-breaker: cam0_md5 → text_thinking entry.
    # Computed lazily — only for questions with multiple candidates.
    tie_lookup = {}  # (q, cam0_md5) -> entry
    needs_tiebreak = {q: ents for q, ents in by_q.items() if len(ents) > 1}
    print(f"[tiebreak] hashing cam0 PNGs for {sum(len(v) for v in needs_tiebreak.values())} text_thinking rows")
    n_hash_fail = 0
    hasher = file_md5_normalized if args.match_mode == "decoded_pixels" else file_md5
    for q, ents in needs_tiebreak.items():
        for e in ents:
            cam0 = remap_path(e["image0_path"])
            if not os.path.exists(cam0):
                n_hash_fail += 1
                continue
            try:
                h = hasher(cam0)
            except Exception:
                n_hash_fail += 1
                continue
            tie_lookup[(q, h)] = e
    print(f"[tiebreak] hashed → {len(tie_lookup)} tie keys ({n_hash_fail} hash failures)")

    bytes_hasher = png_bytes_md5_normalized if args.match_mode == "decoded_pixels" else (
        lambda b: hashlib.md5(b).hexdigest()
    )

    chunk_files = sorted(
        glob.glob(os.path.join(args.visual_only_dir, "*.parquet")),
        key=lambda p: int(re.search(r"chunk_(\d+)\.parquet", os.path.basename(p)).group(1)),
    )
    print(f"[visual_only] {len(chunk_files)} parquet chunks")

    n_total = 0
    n_single = 0
    n_multi_resolved = 0
    n_multi_failed = 0
    n_no_question = 0
    cat_counter = Counter()

    out_image_lists = []
    out_instruction_lists = []
    out_output_text_lists = []
    out_chunk_idx = 0

    unmatched_rows = []  # for Phase 2 dump

    for chunk_path in chunk_files:
        t = pq.read_table(chunk_path)
        chunk_basename = os.path.splitext(os.path.basename(chunk_path))[0]
        for row_idx, (img_list, instr_list, _) in enumerate(zip(
            t.column("image_list").to_pylist(),
            t.column("instruction_list").to_pylist(),
            t.column("output_text_list").to_pylist(),
        )):
            n_total += 1
            q_norm = norm_vo(instr_list[0])
            candidates = by_q.get(q_norm, [])

            chosen = None
            if len(candidates) == 1:
                chosen = candidates[0]
                n_single += 1
            elif len(candidates) > 1:
                cam0_h = bytes_hasher(img_list[0])
                chosen = tie_lookup.get((q_norm, cam0_h))
                if chosen is not None:
                    n_multi_resolved += 1
                else:
                    n_multi_failed += 1
            else:
                n_no_question += 1

            if chosen is None:
                unmatched_rows.append({
                    "id": f"vo_unmatched_{chunk_basename}_{row_idx}",
                    "instruction": instr_list[0],
                    "vo_chunk": chunk_basename,
                    "vo_row": row_idx,
                    "vo_answer_text": "",  # filled below if we can extract it
                })
                continue

            cat_counter[chosen["cat"]] += 1
            think = chosen["think"]
            answer = chosen["answer"]
            out_text_list = [
                f"<think>{think}</think><image_start>",
                f"<image_end><answer>{answer}</answer>",
            ]
            out_image_lists.append(img_list)
            out_instruction_lists.append(instr_list)
            out_output_text_lists.append(out_text_list)

            if len(out_image_lists) >= args.rows_per_chunk:
                p = write_chunk(
                    args.out_dir, out_chunk_idx,
                    out_image_lists, out_instruction_lists, out_output_text_lists,
                )
                print(f"  wrote {p} ({len(out_image_lists)} rows)")
                out_image_lists, out_instruction_lists, out_output_text_lists = [], [], []
                out_chunk_idx += 1

    if out_image_lists:
        p = write_chunk(
            args.out_dir, out_chunk_idx,
            out_image_lists, out_instruction_lists, out_output_text_lists,
        )
        print(f"  wrote {p} ({len(out_image_lists)} rows)")
        out_chunk_idx += 1

    # Re-attach the original visual_only answer (parsed from output_text_list[1])
    # to unmatched rows so Phase 2 can use it as ground truth.
    if unmatched_rows:
        # Map back: for each unmatched row, fetch original output_text_list[1] and parse <answer>X</answer>.
        unmatched_by_chunk = defaultdict(list)
        for u in unmatched_rows:
            unmatched_by_chunk[u["vo_chunk"]].append(u)
        for chunk_basename, group in unmatched_by_chunk.items():
            chunk_path = os.path.join(args.visual_only_dir, f"{chunk_basename}.parquet")
            t = pq.read_table(chunk_path, columns=["output_text_list"])
            outs = t.column("output_text_list").to_pylist()
            for u in group:
                tail = outs[u["vo_row"]][1] if len(outs[u["vo_row"]]) > 1 else ""
                m = ANSWER_RE.search(tail)
                u["vo_answer_text"] = m.group(1).strip() if m else ""

        os.makedirs(os.path.dirname(args.unmatched_jsonl), exist_ok=True)
        with open(args.unmatched_jsonl, "w") as f:
            for u in unmatched_rows:
                f.write(json.dumps(u) + "\n")

    print()
    print("=== Summary ===")
    print(f"  visual_only rows seen           : {n_total}")
    print(f"  matched (single candidate)      : {n_single}")
    print(f"  matched (multi → cam0 tie-break): {n_multi_resolved}")
    print(f"  unmatched (multi, hash miss)    : {n_multi_failed}")
    print(f"  unmatched (no question hit)     : {n_no_question}")
    print(f"  total written                   : {n_single + n_multi_resolved}")
    print(f"  total unmatched                 : {n_multi_failed + n_no_question}")
    print(f"  per-category counts             : {dict(cat_counter)}")
    print(f"  output chunks                   : {out_chunk_idx} → {args.out_dir}")
    print(f"  unmatched dump                  : {args.unmatched_jsonl}")


if __name__ == "__main__":
    main()
