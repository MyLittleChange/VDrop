#!/usr/bin/env python3
"""
HTML audit for the Two-Reader Informativeness oracle renders.

Picks 25 deterministic samples per COSMIC subtask (100 total). Each row
shows the question + gold + V1, V2, and every available oracle view T
side-by-side, so we can eyeball quality before launching the eval.

Images are inlined as base64 data URIs so the HTML is self-contained
(works when opened from a local browser without server access).
"""
import argparse
import base64
import hashlib
import json
import random
from html import escape
from pathlib import Path

INDEX_PATH = Path(
    "/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/two_reader_informativeness_index.json"
)
TEST_JSONS = {
    "anchor": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
    "counting": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
    "relative_distance": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
    "relative_direction": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
}


def img_data_uri(path, max_width=600, jpeg_quality=72):
    """Resize to max_width, encode as JPEG, return data URI."""
    if not path or not Path(path).exists():
        return None
    p = Path(path)
    try:
        from PIL import Image
        from io import BytesIO
        im = Image.open(p).convert("RGB")
        w, h = im.size
        if w > max_width:
            new_w = max_width
            new_h = max(1, int(h * max_width / w))
            im = im.resize((new_w, new_h))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        raw = buf.getvalue()
        ext = "jpeg"
    except Exception:
        raw = p.read_bytes()
        ext = p.suffix.lstrip(".").lower() or "png"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/{ext};base64,{b64}"


def load_question_for(sample):
    asking = sample.get("asking_to", "agent_2")
    if asking == "agent_1":
        q = sample.get("user_1_question", "") or ""
        opts = sample.get("options_user_1") or []
        gold_idx = sample.get("user_1_gt_answer_idx")
        gold_text = sample.get("user_1_gt_answer_text", "") or ""
    else:
        q = sample.get("user_2_question", "") or ""
        opts = sample.get("options_user_2") or []
        gold_idx = sample.get("user_2_gt_answer_idx")
        gold_text = sample.get("user_2_gt_answer_text", "") or ""
    gold_letter = chr(ord("A") + int(gold_idx)) if gold_idx is not None else "?"
    return q, opts, gold_letter, gold_text, asking


def cell(title, src, caption=""):
    if src is None:
        return (
            f'<div class="cell"><div class="title">{escape(title)}</div>'
            f'<div class="missing">— missing —</div>'
            f'<div class="caption">{escape(caption)}</div></div>'
        )
    return (
        f'<div class="cell"><div class="title">{escape(title)}</div>'
        f'<img src="{src}" alt="{escape(title)}"/>'
        f'<div class="caption">{escape(caption)}</div></div>'
    )


def text_cell(title, text):
    if not text:
        return (
            f'<div class="cell"><div class="title">{escape(title)}</div>'
            f'<div class="missing">— pending —</div></div>'
        )
    return (
        f'<div class="cell"><div class="title">{escape(title)}</div>'
        f'<div class="cot">{escape(text)}</div></div>'
    )


def render_row(sample_meta, idx_row, by_subtask_question):
    sid = idx_row["sample_id"]
    subtask = idx_row["subtask"]
    scene = idx_row["scene_id"]
    sample = by_subtask_question[(subtask, sid)]
    q, opts, gold_letter, gold_text, asking = load_question_for(sample)

    opts_parts = []
    for i, opt in enumerate(opts):
        letter = chr(ord("A") + i)
        cls = ' class="gold"' if letter == gold_letter else ""
        opts_parts.append(f"<li{cls}>{escape(letter)}) {escape(str(opt))}</li>")
    opts_html = "".join(opts_parts)

    v1 = img_data_uri(sample.get("user_1_image_local_path"))
    v2 = img_data_uri(sample.get("user_2_image_local_path"))
    td = img_data_uri(idx_row.get("T_td"))
    td_blender = img_data_uri(idx_row.get("T_td_blender"))
    td_gemini = img_data_uri(idx_row.get("T_td_gemini"))
    pano = img_data_uri(idx_row.get("T_pano"))
    pano_gemini = img_data_uri(idx_row.get("T_pano_gemini"))
    cor = idx_row.get("T_cor")
    cor1 = img_data_uri(cor[0]) if cor else None
    cor2 = img_data_uri(cor[1]) if cor else None
    cor_gem = idx_row.get("T_cor_gemini")
    cor_gem1 = img_data_uri(cor_gem[0]) if cor_gem else None
    cor_gem2 = img_data_uri(cor_gem[1]) if cor_gem else None
    noise = img_data_uri(idx_row.get("T_noise"))
    cot = idx_row.get("T_cot") or ""

    cells = [
        cell("V1 (input)", v1, f"asking={asking}"),
        cell("V2 (input)", v2, ""),
        cell("T_td_blender (Cycles ortho)", td_blender, "GT"),
        cell("T_td (synth BEV fallback)", td, "GT"),
        cell("T_td_gemini", td_gemini, "Gemini 3 Pro Image"),
        cell("T_pano (panorama)", pano, "GT"),
        cell("T_pano_gemini", pano_gemini, "Gemini 3 Pro Image"),
        cell("T_cor V1 (point matches)", cor1, "GT"),
        cell("T_cor V2 (point matches)", cor2, "GT"),
        cell("T_cor V1 (Gemini)", cor_gem1, "Gemini 3 Pro Image"),
        cell("T_cor V2 (Gemini)", cor_gem2, "Gemini 3 Pro Image"),
        cell("T_noise (different scene)", noise, ""),
        text_cell("T_cot (LLM CoT)", cot),
    ]

    return f"""
    <div class="row">
      <div class="meta">
        <div class="sid">{escape(sid)} <span class="subtask">[{escape(subtask)}]</span> <span class="scene">scene={escape(scene)}</span></div>
        <div class="q"><b>Q:</b> {escape(q)}</div>
        <ul class="opts">{opts_html}</ul>
        <div class="gold">gold = <b>{escape(gold_letter)}</b>) {escape(gold_text)}</div>
      </div>
      <div class="grid">
        {"".join(cells)}
      </div>
    </div>
    """


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/oracle_audit_100.html",
    )
    parser.add_argument("--per_subtask", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"reading provenance index from {INDEX_PATH}")
    idx = json.load(open(INDEX_PATH))
    samples = idx["samples"]
    summary = idx["summary"]

    # Group by subtask, seeded random selection
    by_subtask = {}
    for r in samples:
        by_subtask.setdefault(r["subtask"], []).append(r)

    rng = random.Random(args.seed)
    chosen = []
    for subtask in ("anchor", "counting", "relative_distance", "relative_direction"):
        pool = by_subtask.get(subtask, [])
        # Prefer samples with as many oracle views as possible
        def n_views(r):
            return sum(bool(r.get(k)) for k in (
                "T_td", "T_td_blender", "T_pano", "T_cor", "T_cot", "T_noise",
                "T_td_gemini", "T_pano_gemini", "T_cor_gemini",
            ))
        pool_sorted = sorted(pool, key=lambda r: (-n_views(r), r["sample_id"]))
        # Deterministic-but-varied: shuffle the top half, take first per_subtask
        top = pool_sorted[: max(args.per_subtask * 4, args.per_subtask)]
        rng.shuffle(top)
        chosen.extend(top[: args.per_subtask])

    print(f"selected {len(chosen)} samples")

    # Look up the original question rows once (avoid re-loading per row)
    by_subtask_question = {}
    for subtask, jp in TEST_JSONS.items():
        for s in json.load(open(jp)):
            by_subtask_question[(subtask, s["sample_id"])] = s

    # Render
    print("rendering rows (this may take ~30-60s for 100 samples)...")
    rows_html = []
    for i, r in enumerate(chosen, 1):
        rows_html.append(render_row(None, r, by_subtask_question))
        if i % 20 == 0:
            print(f"  rendered {i}/{len(chosen)}")

    coverage_html = (
        "<table class='cov'><tr>"
        "<th>Subtask</th>"
        "<th>T_td_blender</th><th>T_td (synth)</th><th>T_td_gemini</th>"
        "<th>T_pano</th><th>T_pano_gemini</th>"
        "<th>T_cor</th><th>T_cor_gemini</th>"
        "<th>T_cot</th><th>T_noise</th>"
        "</tr>"
    )
    for st, v in summary["coverage_per_subtask"].items():
        coverage_html += (
            f"<tr><td>{st}</td>"
            f"<td>{v.get('T_td_blender','—')}</td><td>{v['T_td']}</td><td>{v.get('T_td_gemini','—')}</td>"
            f"<td>{v['T_pano']}</td><td>{v.get('T_pano_gemini','—')}</td>"
            f"<td>{v['T_cor']}</td><td>{v.get('T_cor_gemini','—')}</td>"
            f"<td>{v['T_cot']}</td><td>{v['T_noise']}</td></tr>"
        )
    coverage_html += "</table>"

    page = f"""<!doctype html>
<html><head>
<meta charset="utf-8"/>
<title>Two-Reader Informativeness — Oracle Audit ({len(chosen)} samples)</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1800px; margin: 1rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  .cov {{ border-collapse: collapse; margin: 1rem 0; }}
  .cov th, .cov td {{ border: 1px solid #ccc; padding: 4px 10px; font-size: 0.85rem; text-align: center; }}
  .row {{ border-top: 2px solid #888; padding: 12px 0; }}
  .meta {{ font-size: 0.9rem; margin-bottom: 8px; }}
  .sid {{ font-family: monospace; font-weight: bold; }}
  .subtask {{ background: #eef; padding: 1px 6px; border-radius: 3px; font-weight: normal; }}
  .scene {{ color: #888; font-weight: normal; font-size: 0.8rem; margin-left: 8px; }}
  .q {{ margin: 4px 0; max-width: 1200px; }}
  ul.opts {{ margin: 4px 0 4px 1.5em; padding: 0; }}
  ul.opts li.gold {{ color: #1a7f37; font-weight: 600; }}
  .meta .gold {{ color: #1a7f37; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, minmax(280px, 1fr)); gap: 8px; }}
  .cell {{ background: #fafafa; padding: 6px; border: 1px solid #ddd; border-radius: 4px; }}
  .cell .title {{ font-size: 0.8rem; font-weight: bold; color: #444; margin-bottom: 4px; }}
  .cell img {{ width: 100%; height: auto; display: block; border-radius: 2px; }}
  .cell .caption {{ font-size: 0.75rem; color: #888; margin-top: 2px; }}
  .cell .missing {{ font-size: 0.85rem; color: #b66; padding: 24px 0; text-align: center; background: #fff3f3; border-radius: 2px; }}
  .cell .cot {{ font-size: 0.8rem; max-height: 200px; overflow-y: auto; white-space: pre-wrap; background: #fff; padding: 4px; border-radius: 2px; }}
</style></head><body>
<h1>Two-Reader Informativeness — Oracle Audit ({len(chosen)} samples, {args.per_subtask}/subtask)</h1>
<p>Visual sanity check on the rendered oracle views. Goal: confirm <b>T_td_blender</b> (photoreal Cycles ortho) and <b>T_td</b> (matplotlib BEV fallback) match the room layout in V1/V2, <b>T_pano</b> shows the same scene, <b>T_cor</b> dots mark the same real-world object across V1 and V2 (matching colours), <b>T_noise</b> is clearly a different scene, and <b>T_cot</b> reasoning references the right objects.</p>
<h3>Coverage</h3>
{coverage_html}
{"".join(rows_html)}
</body></html>"""

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"wrote {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
