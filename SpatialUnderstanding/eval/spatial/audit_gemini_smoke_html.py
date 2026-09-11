#!/usr/bin/env python3
"""
Build a small HTML audit comparing Gemini-generated oracles to GT oracles
for the 4-sample smoke run.

Each row shows:
    V1 | V2 | T_td_blender (GT) | T_td_gemini
                | T_pano (GT)   | T_pano_gemini
                | T_cor V1 (GT) | T_cor V2 (GT) | T_cor V1 (Gem) | T_cor V2 (Gem)
"""
import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

GEMINI_ROOT = Path(
    "/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/gemini_oracles"
)
PROVENANCE = Path(
    "/path/to/scratch/VisualCoT/infinigen/"
    "two_reader_informativeness/two_reader_informativeness_index.json"
)
TEST_JSONS = {
    "anchor": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
    "counting": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
    "relative_distance": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
    "relative_direction": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
}

SMOKE_SAMPLE_IDS = [
    "anchor_001984",
    "counting_002443",
    "relative_distance_006213",
    "spatial_003869",  # relative_direction
]


def img_data_uri(path, max_width=600, jpeg_quality=72):
    if not path or not Path(path).exists():
        return None
    p = Path(path)
    try:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        if w > max_width:
            new_w = max_width
            new_h = max(1, int(h * max_width / w))
            im = im.resize((new_w, new_h))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def cell(label, src, sub=""):
    if not src:
        return f'<div class="cell missing"><div class="lab">{label}</div><div class="sub">{sub}</div><div class="ph">— missing —</div></div>'
    return f'<div class="cell"><div class="lab">{label}</div><div class="sub">{sub}</div><img src="{src}"/></div>'


def main():
    idx = json.load(open(PROVENANCE))
    by_sid = {r["sample_id"]: r for r in idx["samples"]}
    raw_by_sid = {}
    for st, jp in TEST_JSONS.items():
        for x in json.load(open(jp)):
            raw_by_sid[x["sample_id"]] = x

    rows_html = []
    for sid in SMOKE_SAMPLE_IDS:
        prov = by_sid.get(sid, {})
        raw = raw_by_sid.get(sid, {})
        scene = prov.get("scene_id", "?")
        subtask = prov.get("subtask", "?")

        v1 = img_data_uri(raw.get("user_1_image_local_path"))
        v2 = img_data_uri(raw.get("user_2_image_local_path"))
        td_blender = img_data_uri(prov.get("T_td_blender"))
        td_synth = img_data_uri(prov.get("T_td"))
        pano_gt = img_data_uri(prov.get("T_pano"))
        cor_gt = prov.get("T_cor") or [None, None]
        cor_gt_v1 = img_data_uri(cor_gt[0])
        cor_gt_v2 = img_data_uri(cor_gt[1])

        td_gem = img_data_uri(GEMINI_ROOT / "topdown" / scene / f"topdown_gemini_{sid}.png")
        pano_gem = img_data_uri(GEMINI_ROOT / "panorama" / scene / f"panorama_gemini_{sid}.png")
        cor_gem_v1 = img_data_uri(GEMINI_ROOT / "cor" / scene / f"cor_gemini_{sid}_v1.png")
        cor_gem_v2 = img_data_uri(GEMINI_ROOT / "cor" / scene / f"cor_gemini_{sid}_v2.png")

        # Pull question + gold answer for context
        question = raw.get("user_2_question", "")
        options = raw.get("options_user_2") or []
        gold_idx = raw.get("user_2_gt_answer_idx")
        opts_html = ""
        if options and gold_idx is not None:
            for i, opt in enumerate(options):
                tag = "gold" if i == gold_idx else "opt"
                opts_html += f'<span class="{tag}">{chr(65+i)}) {opt}</span> '

        row = f"""
<div class="row">
  <div class="meta">
    <span class="sid">{sid}</span>
    <span class="subtask">{subtask}</span>
    <span class="scene">scene={scene}</span>
  </div>
  <div class="q">Q: {question}</div>
  <div class="opts">{opts_html}</div>
  <div class="grp">
    <h3>Inputs</h3>
    <div class="grid grid-2">
      {cell("V1", v1, "")}
      {cell("V2", v2, "")}
    </div>
  </div>
  <div class="grp">
    <h3>Top-down: GT (real Blender) vs Gemini</h3>
    <div class="grid grid-3">
      {cell("T_td_blender (GT Cycles ortho)", td_blender, "")}
      {cell("T_td (synth BEV)", td_synth, "")}
      {cell("T_td_gemini", td_gem, "Gemini 3 Pro Image")}
    </div>
  </div>
  <div class="grp">
    <h3>Panorama: GT vs Gemini</h3>
    <div class="grid grid-2">
      {cell("T_pano (GT)", pano_gt, "")}
      {cell("T_pano_gemini", pano_gem, "Gemini 3 Pro Image")}
    </div>
  </div>
  <div class="grp">
    <h3>Correspondence dots: GT vs Gemini</h3>
    <div class="grid grid-4">
      {cell("T_cor V1 (GT)", cor_gt_v1, "annotate_shared_objects")}
      {cell("T_cor V2 (GT)", cor_gt_v2, "annotate_shared_objects")}
      {cell("T_cor V1 (Gemini)", cor_gem_v1, "")}
      {cell("T_cor V2 (Gemini)", cor_gem_v2, "")}
    </div>
  </div>
</div>
"""
        rows_html.append(row)

    body = "\n".join(rows_html)
    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8"/>
<title>Gemini Oracle Smoke Audit (4 samples)</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1800px; margin: 1rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ font-size: 1.5rem; }}
  h3 {{ font-size: 1rem; margin: 12px 0 6px; color: #555; }}
  .row {{ border-top: 2px solid #888; padding: 16px 0; }}
  .meta {{ font-size: 0.95rem; margin-bottom: 6px; }}
  .sid {{ font-family: monospace; font-weight: bold; }}
  .subtask {{ background: #eef; padding: 1px 8px; border-radius: 3px; font-weight: normal; margin-left: 8px; }}
  .scene {{ color: #888; font-weight: normal; margin-left: 8px; }}
  .q {{ margin: 4px 0; font-size: 0.95rem; }}
  .opts span {{ font-size: 0.85rem; margin-right: 4px; }}
  .opts .gold {{ background: #cfc; padding: 1px 4px; border-radius: 2px; font-weight: bold; }}
  .grp {{ margin: 12px 0; }}
  .grid {{ display: grid; gap: 8px; }}
  .grid-2 {{ grid-template-columns: 1fr 1fr; }}
  .grid-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
  .grid-4 {{ grid-template-columns: 1fr 1fr 1fr 1fr; }}
  .cell {{ border: 1px solid #ddd; padding: 6px; }}
  .cell.missing {{ background: #fee; }}
  .cell img {{ width: 100%; height: auto; display: block; }}
  .lab {{ font-size: 0.8rem; font-weight: bold; }}
  .sub {{ font-size: 0.75rem; color: #888; margin-bottom: 4px; }}
  .ph {{ color: #c33; font-size: 0.85rem; padding: 30px 0; text-align: center; }}
</style>
</head><body>
<h1>Gemini Oracle Smoke Audit — 4 samples × td/pano/cor</h1>
<p>
  Model: <b>gemini-3-pro-image-preview</b>.
  Cost: ~$1 for 12 generated images.
  All 12 API calls succeeded (3 cor calls returned only 1 image instead of 2).
</p>
<p>
  Compare GT-rendered oracles (left) to Gemini-generated oracles (right). Look for:
  (1) does T_td_gemini look like a top-down or a 3/4 elevated view?
  (2) is T_pano_gemini scene-faithful?
  (3) does T_cor_gemini consistently mark the same object across v1/v2?
</p>
{body}
</body></html>
"""
    out_path = (
        Path("/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness")
        / "gemini_smoke_audit.html"
    )
    out_path.write_text(html)
    sz_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path} ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
