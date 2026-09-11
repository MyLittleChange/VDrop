#!/usr/bin/env python3
"""
Pick one random fully-covered sample per subtask and render an HTML page
showing V1, V2, and every oracle image (T_pano, T_cor, T_td_blender,
T_noise, plus the 3 model-generated *_gen variants).

Output: a single self-contained HTML file with all images base64-embedded.
"""
import argparse
import base64
import io
import json
import random
from pathlib import Path

from PIL import Image

SUBTASKS = ["anchor", "counting", "relative_distance", "relative_direction"]
ORACLE_KEYS = [
    ("V1", "user_1_image_local_path"),
    ("V2", "user_2_image_local_path"),
    ("T_td_blender (GT)", "_T_td_blender"),
    ("T_td_gen (BAGEL bridge)", "_T_td_gen"),
    ("T_cor composite (GT)", "_T_cor_composite"),
    ("T_cor_gen (BAGEL bridge)", "_T_cor_gen"),
    ("T_pano (GT)", "_T_pano"),
    ("T_pano_gen (BAGEL bridge)", "_T_pano_gen"),
    ("T_noise (random scene pano)", "_T_noise"),
]

DEFAULT_SUBSET = "/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/eval_subset_200"
DEFAULT_OUT = "/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/oracle_examples_one_per_subtask.html"


def img_to_data_uri(path, max_width=640, quality=82):
    if not path or not Path(path).exists():
        return None
    try:
        im = Image.open(path).convert("RGB")
        if im.width > max_width:
            new_h = int(im.height * max_width / im.width)
            im = im.resize((max_width, new_h))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    except Exception as e:
        return None


def cell(label, uri, note=""):
    if uri is None:
        body = '<div class="missing">— missing —</div>'
    else:
        body = f'<img src="{uri}"/>'
    note_html = f'<div class="note">{note}</div>' if note else ''
    return (
        f'<div class="cell">'
        f'<div class="label">{label}</div>'
        f'{body}{note_html}'
        f'</div>'
    )


def render_sample_block(subtask, sample):
    sid = sample["sample_id"]
    scene = sample.get("scene_id", "")
    asking = sample.get("asking_to", "?")
    q = sample.get("user_2_question") or sample.get("user_1_question") or ""
    opts = sample.get("options_user_2") or sample.get("options_user_1") or []
    gold_idx = sample.get("user_2_gt_answer_idx")
    if gold_idx is None:
        gold_idx = sample.get("user_1_gt_answer_idx")
    gold_text = sample.get("user_2_gt_answer_text") or sample.get("user_1_gt_answer_text", "")
    gold_letter = chr(65 + int(gold_idx)) if gold_idx is not None else "?"

    options_html = "<ol type='A'>"
    for i, opt in enumerate(opts):
        letter = chr(65 + i)
        is_gold = (letter == gold_letter)
        cls = "gold" if is_gold else ""
        options_html += f'<li class="{cls}">{opt}</li>'
    options_html += "</ol>"

    cells = []
    for label, key in ORACLE_KEYS:
        path = sample.get(key)
        # Mark the asker-relevant T_cor explicitly
        uri = img_to_data_uri(path)
        cells.append(cell(label, uri))

    return f'''
<section class="sample">
  <h2><span class="subtask">{subtask}</span> {sid} <span class="scene">scene={scene}, asking_to={asking}</span></h2>
  <div class="q"><b>Question:</b> {q}</div>
  <div class="options"><b>Options:</b> {options_html}<div class="gold-answer">Gold: <b>{gold_letter}</b> ({gold_text})</div></div>
  <div class="grid">
    {''.join(cells)}
  </div>
</section>
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_dir", default=DEFAULT_SUBSET)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples_per_subtask", type=int, default=1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    picks = []  # list of (subtask, sample dict)

    for subtask in SUBTASKS:
        d = json.load(open(Path(args.subset_dir) / f"{subtask}.json"))
        # Only consider samples where every oracle key resolves to an existing file
        full = []
        for s in d:
            ok = True
            for _, key in ORACLE_KEYS:
                if key in ("user_1_image_local_path", "user_2_image_local_path"):
                    if not s.get(key) or not Path(s[key]).exists():
                        ok = False; break
                else:
                    if not s.get(key) or not Path(s[key]).exists():
                        ok = False; break
            if ok:
                full.append(s)
        if not full:
            print(f"WARN: no fully-covered samples for {subtask}")
            continue
        chosen = rng.sample(full, min(args.samples_per_subtask, len(full)))
        for s in chosen:
            picks.append((subtask, s))
            print(f"  picked {subtask}/{s['sample_id']} (scene={s['scene_id']})")

    # Render
    body = "".join(render_sample_block(st, s) for (st, s) in picks)
    html = f'''<!doctype html>
<html><head>
<meta charset="utf-8"/>
<title>Oracle examples — one per subtask</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1600px; margin: 1rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; padding-top: 1rem; border-top: 2px solid #888; }}
  .subtask {{ background: #eef; padding: 1px 8px; border-radius: 3px; font-size: 0.85rem; margin-right: 0.5rem; }}
  .scene {{ color: #888; font-weight: normal; font-size: 0.85rem; margin-left: 0.5rem; }}
  .q {{ margin: 0.5rem 0; font-size: 0.95rem; }}
  .options {{ margin: 0.5rem 0 1rem; font-size: 0.9rem; }}
  .options ol {{ margin: 0.3rem 0 0.3rem 1.5rem; }}
  .options li.gold {{ font-weight: bold; color: #060; }}
  .gold-answer {{ margin-top: 0.3rem; color: #060; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
  .cell {{ border: 1px solid #ccc; padding: 6px; background: #fafafa; }}
  .cell .label {{ font-size: 0.85rem; font-weight: bold; margin-bottom: 4px; color: #444; }}
  .cell img {{ width: 100%; height: auto; display: block; }}
  .cell .note {{ font-size: 0.75rem; color: #888; margin-top: 4px; }}
  .missing {{ background: #fee; color: #a00; font-size: 0.85rem; padding: 1rem; text-align: center; }}
  .legend {{ background: #f8f8f8; border: 1px solid #ddd; padding: 0.6rem 1rem; font-size: 0.85rem; margin: 0.5rem 0 1.5rem; }}
  .legend code {{ background: #eee; padding: 1px 4px; border-radius: 2px; }}
</style>
</head>
<body>
<h1>Oracle examples — one random sample per subtask</h1>
<div class="legend">
  Three columns × three rows of oracle images per sample:
  <ul style="margin:0.3rem 0 0 1.2rem;padding:0;">
    <li><b>Row 1</b>: input views (V1, V2) — used by every condition.</li>
    <li><b>Row 2</b>: <b>T_td_blender</b> = GT photoreal Cycles ortho top-down. <b>T_td_gen</b> = BAGEL's <i>generated</i> top-down (from <code>topdown_round3_visual_only_bridge_masked_lora_7k</code>).</li>
    <li><b>Row 3 left/middle</b>: <b>T_cor composite</b> = GT side-by-side V1|V2 with coloured dots on co-visible objects. <b>T_cor_gen</b> = BAGEL's generated correspondence image (from <code>pm_no_rotation_visual_only_bridge_masked_lora_7k</code>).</li>
    <li><b>Row 3 right + Row 4</b>: <b>T_pano</b> = GT 360° panorama. <b>T_pano_gen</b> = BAGEL's generated panorama (from <code>mix_all_balance_visual_only_bridge_masked_lora_7k</code>). <b>T_noise</b> = random unrelated-scene panorama (null/control condition).</li>
  </ul>
</div>
{body}
</body></html>
'''
    out = Path(args.output)
    out.write_text(html)
    sz_mb = out.stat().st_size / 1e6
    print(f"\nWrote {out}  ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
