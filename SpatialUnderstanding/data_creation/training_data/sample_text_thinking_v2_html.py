#!/usr/bin/env python3
"""Random-sample N rows across text_thinking_v2/*.jsonl and render an HTML
viewer (cam0 + cam1 + question + GT + <think> + <answer>) so the user can
spot-check annotation quality.
"""

import argparse
import base64
import glob
import html
import io
import json
import os
import random
import re

from PIL import Image

DEFAULT_DIR = "/path/to/scratch/infinigen/training_data_mix_all_balance/text_thinking_v2"
DEFAULT_OUT = "/path/to/scratch/infinigen/training_data_mix_all_balance/text_thinking_v2_sample_100.html"
IMG_BASE = "/network/scratch/"
THUMB_WIDTH = 640


def img_to_data_uri(path: str) -> str:
    """Open, resize to THUMB_WIDTH, encode to base64 JPEG data URI."""
    im = Image.open(path).convert("RGB")
    if im.width > THUMB_WIDTH:
        new_h = int(im.height * THUMB_WIDTH / im.width)
        im = im.resize((THUMB_WIDTH, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def split_assistant(value: str):
    think = ""
    answer = ""
    m = THINK_RE.search(value)
    if m:
        think = m.group(1).strip()
    m = ANS_RE.search(value)
    if m:
        answer = m.group(1).strip()
    return think, answer


def strip_user(value: str) -> str:
    """Drop leading instruction + <image><image> tags so we just show the question."""
    v = value
    v = re.sub(r"<image>", "", v)
    # remove the boilerplate prefix everything before the first blank line after instructions
    # easier: split off anything up to and including the last "Provide your final answer wrapped..." sentence + tags
    marker = "<answer> answer here </answer>."
    if marker in v:
        v = v.split(marker, 1)[1]
    return v.strip()


def category_of(p: str) -> str:
    return os.path.basename(p).replace(".jsonl", "")


PER_CAT = {"counting", "anchor", "spatial", "relative_distance", "perspective_taking"}


def load_all(d: str):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        cat = category_of(p)
        if cat not in PER_CAT:
            continue  # skip the merged text_thinking.jsonl
        with open(p) as f:
            for ln in f:
                r = json.loads(ln)
                r["_cat"] = cat
                rows.append(r)
    return rows


def img_path(rel: str) -> str:
    rel = rel.lstrip("/")
    return os.path.join(IMG_BASE, rel)


def render_html(samples, out_path: str):
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>text_thinking_v2 sample</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;max-width:1300px;margin:24px auto;padding:0 12px;background:#fafafa;color:#222}",
        ".row{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px;margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,.04)}",
        ".meta{font-size:12px;color:#666;margin-bottom:8px}",
        ".meta b{color:#222}",
        ".imgs{display:flex;gap:10px;margin-bottom:10px}",
        ".imgs figure{margin:0;flex:1}",
        ".imgs img{width:100%;height:auto;border:1px solid #ccc;border-radius:4px;display:block}",
        ".imgs figcaption{font-size:11px;color:#888;text-align:center;margin-top:2px}",
        ".q{background:#f5f7fb;border-left:3px solid #5577cc;padding:8px 10px;margin:8px 0;white-space:pre-wrap;font-size:13px}",
        ".think{background:#fffbe6;border-left:3px solid #d4a017;padding:8px 10px;margin:8px 0;white-space:pre-wrap;font-size:13px;line-height:1.45}",
        ".ans{background:#e6f4ea;border-left:3px solid #2e7d32;padding:6px 10px;margin:8px 0;font-weight:600}",
        ".badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;margin-right:6px}",
        ".b-cat{background:#dde7ff;color:#1a3aa0}",
        ".b-src-v4{background:#fde4e4;color:#a01a1a}",
        ".b-src-v5{background:#e4f0fd;color:#1a4ea0}",
        ".b-src-ankur{background:#e4fde6;color:#1a8025}",
        ".id{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:#444}",
        "h1{font-size:18px;margin:8px 0 18px}",
        "summary{cursor:pointer;font-weight:600;color:#5577cc;margin-top:6px;font-size:12px}",
        "</style></head><body>",
        f"<h1>text_thinking_v2 — {len(samples)} random samples</h1>",
    ]

    for i, r in enumerate(samples, 1):
        sid = r["id"]
        cat = r["_cat"]
        src = "v4" if sid.startswith("v4_") else "v5" if sid.startswith("v5_") else "ankur" if sid.startswith("ankur_") else "?"
        user_v = next((c["value"] for c in r["conversations"] if c["from"] == "human"), "")
        gpt_v = next((c["value"] for c in r["conversations"] if c["from"] == "gpt"), "")
        question = strip_user(user_v)
        think, ans = split_assistant(gpt_v)
        imgs = r.get("image", [])
        cam0 = img_path(imgs[0]) if len(imgs) > 0 else ""
        cam1 = img_path(imgs[1]) if len(imgs) > 1 else ""
        cam0_exists = os.path.exists(cam0)
        cam1_exists = os.path.exists(cam1)

        parts.append("<div class='row'>")
        parts.append(
            f"<div class='meta'>"
            f"<span class='badge b-cat'>{html.escape(cat)}</span>"
            f"<span class='badge b-src-{src}'>{src}</span>"
            f"<span class='id'>{html.escape(sid)}</span>"
            f" &nbsp; <b>#{i}</b>"
            f"</div>"
        )
        parts.append("<div class='imgs'>")
        for idx, (p, ok) in enumerate([(cam0, cam0_exists), (cam1, cam1_exists)], 1):
            cap = f"image {idx}" + ("" if ok else " (MISSING)")
            if ok:
                try:
                    src = img_to_data_uri(p)
                    parts.append(f"<figure><img src='{src}'><figcaption>{html.escape(cap)}</figcaption></figure>")
                except Exception as e:
                    parts.append(f"<figure><div style='padding:20px;background:#fee;color:#900;text-align:center'>load error: {html.escape(str(e))}</div><figcaption>{html.escape(cap)}</figcaption></figure>")
            else:
                parts.append(f"<figure><div style='padding:20px;background:#fee;color:#900;text-align:center'>missing<br><span style='font-size:10px'>{html.escape(p)}</span></div><figcaption>{html.escape(cap)}</figcaption></figure>")
        parts.append("</div>")
        parts.append(f"<div class='q'>{html.escape(question)}</div>")
        parts.append(f"<div class='think'>{html.escape(think) if think else '(no &lt;think&gt;)'}</div>")
        parts.append(f"<div class='ans'>answer: {html.escape(ans) if ans else '(none)'}</div>")
        parts.append(
            f"<details><summary>raw row</summary><pre style='font-size:11px;overflow:auto'>{html.escape(json.dumps(r, indent=2)[:4000])}</pre></details>"
        )
        parts.append("</div>")

    parts.append("</body></html>")
    with open(out_path, "w") as f:
        f.write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=DEFAULT_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_all(args.input_dir)
    print(f"Loaded {len(rows)} rows from {args.input_dir}")
    random.seed(args.seed)
    random.shuffle(rows)
    samples = rows[: args.n]

    # Stratified-ish report
    from collections import Counter
    cat_dist = Counter(r["_cat"] for r in samples)
    src_dist = Counter(
        "v4" if r["id"].startswith("v4_") else "v5" if r["id"].startswith("v5_") else "ankur" if r["id"].startswith("ankur_") else "?"
        for r in samples
    )
    print(f"Sample category dist: {dict(cat_dist)}")
    print(f"Sample source   dist: {dict(src_dist)}")

    render_html(samples, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
