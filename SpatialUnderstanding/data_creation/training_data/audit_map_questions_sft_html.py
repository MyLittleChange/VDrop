#!/usr/bin/env python3
"""
Sample N pos + N neg rows from the 3-mode SFT pairing and render them to a
single self-contained HTML file for manual quality review.

Each row shows 5 thumbnails:
  cam0 (input) | cam1 (input) | target map (input, the one being judged)
                                | panorama (bridge for visual_think)
                                | correct top-down (bridge for visual_think_topdown)

For positive (Yes / A) rows the target map IS the correct top-down — the two
right-most thumbnails will be identical by construction.

Inputs are taken from the SFT pairing.json + the scene_index.json + on-disk
renders; no parquet decoding required.

Usage:
    python audit_map_questions_sft_html.py \
        --pairing /path/to/scratch/VisualCoT/training_data/map_questions_sft/pairing.json \
        --scene_index /path/to/scratch/VisualCoT/training_data/map_questions_synth/scene_index.json \
        --output_html /path/to/scratch/VisualCoT/training_data/map_questions_sft/audit_100.html \
        --n 100
"""

import argparse
import base64
import glob
import html
import json
import os
import random


PANORAMA_ROOTS = [
    "/path/to/scratch/VisualCoT/infinigen/rendered_panorama_train",
    "/path/to/scratch/infinigen/rendered_panorama_v4",
    "/path/to/scratch/infinigen/rendered_panorama_v5",
]
PERTURBED_ROOT = "/path/to/scratch/infinigen/map_questions_perturbed"


def remap_path(path):
    if path is None:
        return None
    path = path.replace("/path/to/scratch", "/path/to/scratch")
    path = path.replace("/path/to/scratch", "/path/to/scratch")
    return path


def find_panorama(scene_id):
    for root in PANORAMA_ROOTS:
        d = os.path.join(root, scene_id)
        if not os.path.isdir(d):
            continue
        hits = glob.glob(os.path.join(d, "panorama_blender_limits_*.png"))
        if hits:
            return sorted(hits)[0]
    return None


def correct_topdown_path(scene_id, asking_to):
    n = 1 if asking_to == "agent_1" else 2
    return os.path.join(PERTURBED_ROOT, scene_id, f"correct_agent_{n}.png")


def resolve_neg_target(scene_id, asking_to, sid):
    """For a neg sid, deterministic-replay the same RNG used by the SFT script:
    rng = Random(hash((scene_id, asking_to, 0)) & 0xFFFFFFFF); rng.choice(perturbed_maps)."""
    idx_path = os.path.join(PERTURBED_ROOT, scene_id, "index.json")
    if not os.path.exists(idx_path):
        return None
    try:
        idx = json.load(open(idx_path))
    except Exception:
        return None
    ag = idx.get("agents", {}).get(asking_to)
    if not ag or not ag.get("perturbed_maps"):
        return None
    pert_maps = [pm for pm in ag["perturbed_maps"] if os.path.exists(pm["path"])]
    if not pert_maps:
        return None
    rng = random.Random(hash((scene_id, asking_to, 0)) & 0xFFFFFFFF)
    return rng.choice(pert_maps)["path"]


def img_to_data_uri(path, max_dim=512, jpeg_quality=75):
    if not path or not os.path.exists(path):
        return ""
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


def img_block(uri, label):
    if uri:
        return f'<div class="cell"><div class="lbl">{label}</div><img src="{uri}" /></div>'
    return f'<div class="cell"><div class="lbl">{label}</div><div class="missing">missing</div></div>'


def render_row(idx, sid, base_row, label):
    scene_id = base_row["scene_id"]
    asking_to = base_row.get("asking_to", "agent_2")
    source_root = base_row.get("source_root", "")
    cam0 = remap_path(base_row["user_1_image_local_path"])
    cam1 = remap_path(base_row["user_2_image_local_path"])
    correct_top = correct_topdown_path(scene_id, asking_to)
    panorama = find_panorama(scene_id)

    if label == "pos":
        target_map = correct_top
        ans_letter, ans_text = "A", "Yes"
        ans_class = "ans-yes"
    else:
        target_map = resolve_neg_target(scene_id, asking_to, sid)
        ans_letter, ans_text = "B", "No"
        ans_class = "ans-no"

    question = html.escape(base_row.get("question_both_views") or base_row.get("question") or "")

    sid_e = html.escape(sid)
    scene_e = html.escape(scene_id)
    room_e = html.escape(base_row.get("room_part", ""))
    asking_e = html.escape(asking_to)

    return f"""
<div class="row">
  <div class="meta">
    <span class="idx">#{idx}</span>
    <span class="{ans_class}">answer: {ans_letter} ({ans_text})</span>
    <span>scene: <code>{scene_e}</code></span>
    <span>room_part: <code>{room_e}</code></span>
    <span>asking_to: <code>{asking_e}</code></span>
    <span>source: <code>{html.escape(source_root)}</code></span>
  </div>
  <div class="qline">{question}</div>
  <div class="five">
    {img_block(img_to_data_uri(cam0), "cam0 (input)")}
    {img_block(img_to_data_uri(cam1), "cam1 (input)")}
    {img_block(img_to_data_uri(target_map), "target map (input)")}
    {img_block(img_to_data_uri(panorama, max_dim=768), "panorama bridge (visual_think)")}
    {img_block(img_to_data_uri(correct_top), "correct top-down bridge (visual_think_topdown)")}
  </div>
  <div class="sid"><code>{sid_e}</code></div>
</div>
"""


HEAD = """<!doctype html>
<html><head><meta charset="utf-8"><title>Map Questions 3-mode SFT Audit</title>
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
  .five { display: grid; grid-template-columns: 1fr 1fr 1fr 1.5fr 1fr; gap: 10px; align-items: end; }
  .cell { display: flex; flex-direction: column; align-items: center; }
  .cell img { max-width: 100%; max-height: 280px; border: 1px solid #ccc; border-radius: 4px; }
  .lbl { font-size: 11px; color: #666; margin-bottom: 4px; text-align: center; }
  .missing { width: 100%; height: 200px; display: flex; align-items: center; justify-content: center; border: 1px dashed #f00; color: #f00; }
  .sid { margin-top: 8px; font-size: 11px; color: #888; }
  code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 12px; }
</style>
</head><body>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairing", default="/path/to/scratch/VisualCoT/training_data/map_questions_sft/pairing.json")
    p.add_argument("--scene_index", default="/path/to/scratch/VisualCoT/training_data/map_questions_synth/scene_index.json")
    p.add_argument("--output_html", required=True)
    p.add_argument("--n", type=int, default=100, help="Total samples; balanced 50/50 pos/neg.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    pairing = json.load(open(args.pairing))
    sids = pairing["sample_ids_in_order"]
    print(f"Loaded {len(sids)} sids from pairing.json")

    scene_index = json.load(open(args.scene_index))
    scenes = scene_index["scenes"] if isinstance(scene_index, dict) else scene_index
    by_scene = {r["scene_id"]: r for r in scenes}

    rng = random.Random(args.seed)
    pos_sids = [s for s in sids if s.endswith("_pos")]
    neg_sids = [s for s in sids if s.endswith("_neg")]
    n_pos = min(args.n // 2, len(pos_sids))
    n_neg = min(args.n - n_pos, len(neg_sids))
    sampled = ([(s, "pos") for s in rng.sample(pos_sids, n_pos)] +
               [(s, "neg") for s in rng.sample(neg_sids, n_neg)])
    rng.shuffle(sampled)
    print(f"Sampling {len(sampled)} ({n_pos} pos + {n_neg} neg)")

    parts = [HEAD]
    parts.append(f"<h1>Map Questions 3-mode SFT Audit — {len(sampled)} samples</h1>")
    parts.append(f'<div class="summary">Pairing: <code>{html.escape(args.pairing)}</code> '
                 f'· total_sids: {len(sids)} · seed={args.seed}'
                 f'<br/>Scene index: <code>{html.escape(args.scene_index)}</code> ({len(scenes)} scenes)'
                 f'<br/>Each row has 3 input images (cam0, cam1, target map) + the two bridge images '
                 f'used by the visual_think and visual_think_topdown modes.</div>')

    for i, (sid, label) in enumerate(sampled):
        # Recover the scene_id from sid: sid = base_scene_..._{pos|neg}
        # The base sample_id in synth includes the scene 8-char hex; easier to
        # find the synth row whose sample_id startswith the same prefix, but we
        # cached by scene_id, and sid carries scene_id as token-1 after
        # 'map_synth_'. Use a substring lookup over by_scene keys.
        scene_id = None
        for s in by_scene:
            if s in sid:
                scene_id = s
                break
        if scene_id is None:
            print(f"  [WARN] could not resolve scene_id for sid={sid}")
            continue
        base_row = by_scene[scene_id]
        parts.append(render_row(i, sid, base_row, label))
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
