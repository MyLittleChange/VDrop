#!/usr/bin/env python3
"""
Sample N synthesized map questions and render them to a single HTML file for
manual quality review. Each row shows cam0 | cam1 | topdown (the bridge map),
plus metadata (sample_id, scene_id, room_part, distractor_scene_id, answer).

Images are base64-embedded so the HTML is self-contained and can be opened
on any machine after `scp`-ing the single .html file.

Usage:
    python audit_map_questions_html.py \
        --input_json /path/to/scratch/VisualCoT/training_data/map_questions_synth/map_questions_synth.json \
        --output_html /tmp/map_questions_audit_100.html \
        --n 100 --seed 0
"""

import argparse
import base64
import html
import json
import os
import random


def img_to_data_uri(path: str, max_dim: int = 512, jpeg_quality: int = 75) -> str:
    """Read an image, downsize to <= max_dim on the long side, return JPEG data URI."""
    try:
        from io import BytesIO
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if max(w, h) > max_dim:
                if w >= h:
                    nw, nh = max_dim, int(h * max_dim / w)
                else:
                    nw, nh = int(w * max_dim / h), max_dim
                im = im.resize((nw, nh), Image.LANCZOS)
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=jpeg_quality)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


def render_row(idx: int, sample: dict) -> str:
    sid = html.escape(sample["sample_id"])
    scene = html.escape(sample.get("scene_id", ""))
    room = html.escape(sample.get("room_part", ""))
    distractor = sample.get("distractor_scene_id") or "—"
    distractor = html.escape(str(distractor))
    answer_text = html.escape(sample.get("user_2_gt_answer_text", ""))
    answer_idx = sample.get("correct_index", "")
    answer_letter = chr(65 + int(answer_idx)) if answer_idx in (0, 1) else "?"
    answer_class = "ans-yes" if answer_idx == 0 else "ans-no"
    question = html.escape(sample.get("question_both_views", ""))

    cam0_uri = img_to_data_uri(sample["user_1_image_local_path"])
    cam1_uri = img_to_data_uri(sample["user_2_image_local_path"])
    map_uri = img_to_data_uri(sample["map_image_path"])

    def img_block(uri, label):
        if uri:
            return f'<div class="cell"><div class="lbl">{label}</div><img src="{uri}" /></div>'
        return f'<div class="cell"><div class="lbl">{label}</div><div class="missing">missing</div></div>'

    return f"""
<div class="row">
  <div class="meta">
    <span class="idx">#{idx}</span>
    <span class="{answer_class}">answer: {answer_letter} ({answer_text})</span>
    <span>scene: <code>{scene}</code></span>
    <span>room_part: <code>{room}</code></span>
    <span>distractor_scene: <code>{distractor}</code></span>
  </div>
  <div class="qline">{question}</div>
  <div class="trio">
    {img_block(cam0_uri, "cam0 (user_1)")}
    {img_block(cam1_uri, "cam1 (user_2)")}
    {img_block(map_uri, "top-down map (asked about)")}
  </div>
  <div class="sid"><code>{sid}</code></div>
</div>
"""


HEAD = """<!doctype html>
<html><head><meta charset="utf-8"><title>Map Questions Audit</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 24px; background: #fafafa; }
  h1 { margin: 0 0 8px 0; }
  .summary { color: #555; margin-bottom: 24px; }
  .row { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 14px 16px; margin-bottom: 18px; }
  .meta { font-size: 13px; color: #333; display: flex; gap: 16px; flex-wrap: wrap; align-items: center; margin-bottom: 6px; }
  .meta .idx { font-weight: 700; color: #888; }
  .ans-yes { background: #d6f5d6; color: #14532d; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
  .ans-no  { background: #fde2e2; color: #7f1d1d; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
  .qline { font-size: 14px; color: #222; margin: 4px 0 10px 0; font-style: italic; }
  .trio { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
  .cell { display: flex; flex-direction: column; align-items: center; }
  .cell img { max-width: 100%; max-height: 320px; border: 1px solid #ccc; border-radius: 4px; }
  .lbl { font-size: 12px; color: #666; margin-bottom: 4px; }
  .missing { width: 100%; height: 200px; display: flex; align-items: center; justify-content: center; border: 1px dashed #f00; color: #f00; }
  .sid { margin-top: 8px; font-size: 11px; color: #888; }
  code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 12px; }
</style>
</head><body>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_json", required=True)
    p.add_argument("--output_html", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data = json.load(open(args.input_json))
    print(f"Loaded {len(data)} synthesized samples from {args.input_json}")

    rng = random.Random(args.seed)
    # Sample 50/50 pos/neg if possible so reviewer sees both classes
    pos = [s for s in data if s.get("correct_index") == 0]
    neg = [s for s in data if s.get("correct_index") == 1]
    n_pos = min(args.n // 2, len(pos))
    n_neg = min(args.n - n_pos, len(neg))
    sampled = rng.sample(pos, n_pos) + rng.sample(neg, n_neg)
    rng.shuffle(sampled)
    print(f"Sampling {len(sampled)} ({n_pos} pos + {n_neg} neg)")

    parts = [HEAD]
    parts.append(f"<h1>Map Questions Audit — {len(sampled)} samples</h1>")
    parts.append(f'<div class="summary">Source: <code>{html.escape(args.input_json)}</code> '
                 f'· dataset size: {len(data)} · sampled with seed={args.seed}</div>')
    for i, s in enumerate(sampled):
        parts.append(render_row(i, s))
        if (i + 1) % 10 == 0:
            print(f"  rendered {i+1}/{len(sampled)}")
    parts.append("</body></html>")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_html)), exist_ok=True)
    with open(args.output_html, "w") as f:
        f.write("".join(parts))
    sz = os.path.getsize(args.output_html) / 1e6
    print(f"Wrote {args.output_html} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
