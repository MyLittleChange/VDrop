"""Load VSI-Bench inference results and generate a self-contained HTML page
with embedded input frames, generated thinking images, and evaluation details."""

import argparse
import base64
import glob
import html
import json
import os
import random
from collections import defaultdict
from io import BytesIO

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Frame extraction (mirrors inference: deterministic np.linspace sampling)
# ---------------------------------------------------------------------------

def sample_frames_from_video(video_path: str, n_frames: int = 8):
    """Extract n_frames uniformly from an mp4 using decord (same as inference)."""
    import decord
    vr = decord.VideoReader(video_path, num_threads=1)
    frame_indices = np.linspace(0, len(vr) - 1, n_frames, dtype=int).tolist()
    frames = vr.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(f).convert("RGB") for f in frames]


def img_to_base64(img, thumb_size=(256, 256)):
    """Resize PIL image to thumbnail and return base64 data URI."""
    img = img.copy()
    img.thumbnail(thumb_size)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def file_to_base64(path, thumb_size=(256, 256)):
    """Load image file, thumbnail, return base64 data URI."""
    img = Image.open(path).convert("RGB")
    return img_to_base64(img, thumb_size)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

NUMERIC_ANSWER_TYPES = {
    "object_counting",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
}


def load_results(results_dir: str):
    """Load and merge all shard results from a directory."""
    combined = os.path.join(results_dir, "inference_results_combined.json")
    if os.path.exists(combined):
        with open(combined) as f:
            data = json.load(f)
        return data.get("results", []), data.get("config", {}), data.get("metrics", {})

    shard_files = sorted(glob.glob(os.path.join(results_dir, "inference_results_shard*.json")))
    # Exclude checkpoint files
    shard_files = [f for f in shard_files if "checkpoint" not in f]
    if not shard_files:
        raise FileNotFoundError(f"No inference result files found in {results_dir}")

    all_results = {}
    config = {}
    for sf in shard_files:
        with open(sf) as f:
            data = json.load(f)
        if not config:
            config = data.get("config", {})
        for r in data.get("results", []):
            sid = r.get("sample_id", "")
            if sid and r.get("final_answer_text", "").strip():
                all_results[sid] = r

    results = list(all_results.values())
    return results, config, {}


def stratified_sample(results, num_samples, seed):
    """Sample up to num_samples, stratified by question_type."""
    random.seed(seed)
    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)

    n_types = len(by_type)
    per_type = max(1, num_samples // n_types)
    sampled = []
    for qt in sorted(by_type.keys()):
        pool = by_type[qt]
        random.shuffle(pool)
        sampled.extend(pool[:per_type])

    # Fill remaining quota
    remaining = num_samples - len(sampled)
    if remaining > 0:
        sampled_ids = {r["sample_id"] for r in sampled}
        leftovers = [r for r in results if r["sample_id"] not in sampled_ids]
        random.shuffle(leftovers)
        sampled.extend(leftovers[:remaining])

    return sampled[:num_samples]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_summary(results):
    """Compute summary stats from results list."""
    letter = [r for r in results if not r.get("is_numeric") and r.get("accuracy") is not None]
    numeric = [r for r in results if r.get("is_numeric") and r.get("mra") is not None]

    overall_acc = sum(r["accuracy"] for r in letter) / len(letter) if letter else None
    overall_mra = sum(r["mra"] for r in numeric) / len(numeric) if numeric else None

    per_type = defaultdict(lambda: {"scores": [], "metric": None})
    for r in results:
        qt = r["question_type"]
        if r.get("is_numeric"):
            per_type[qt]["metric"] = "mra"
            if r.get("mra") is not None:
                per_type[qt]["scores"].append(r["mra"])
        else:
            per_type[qt]["metric"] = "accuracy"
            if r.get("accuracy") is not None:
                per_type[qt]["scores"].append(r["accuracy"])

    type_stats = {}
    for qt, info in sorted(per_type.items()):
        scores = info["scores"]
        type_stats[qt] = {
            "metric": info["metric"],
            "value": sum(scores) / len(scores) if scores else 0,
            "count": len(scores),
        }

    return {
        "overall_accuracy": overall_acc,
        "overall_mra": overall_mra,
        "total_letter": len(letter),
        "total_numeric": len(numeric),
        "per_type": type_stats,
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def score_cell(result):
    """Return (display_text, css_class) for a result's score."""
    if result.get("is_numeric"):
        mra = result.get("mra", 0)
        color = "green" if mra and mra >= 0.5 else "red"
        return f"MRA: {mra:.2f}" if mra is not None else "N/A", color
    else:
        acc = result.get("accuracy", 0)
        color = "green" if acc == 1.0 else "red"
        return ("Correct" if acc == 1.0 else "Wrong"), color


def build_html(results, summary, config, data_dir, num_frames, thumb_size, generated_images_dir):
    """Build the full HTML string."""
    total = summary["total_letter"] + summary["total_numeric"]

    # Summary table
    acc_str = f"{summary['overall_accuracy']:.4f}" if summary["overall_accuracy"] is not None else "N/A"
    mra_str = f"{summary['overall_mra']:.4f}" if summary["overall_mra"] is not None else "N/A"
    model_name = os.path.basename(config.get("model_path", "unknown"))
    thinking_mode = config.get("thinking_mode", "unknown")

    type_rows = ""
    for qt, info in sorted(summary["per_type"].items()):
        type_rows += (
            f'<tr><td>{html.escape(qt)}</td>'
            f'<td>{info["metric"]}</td>'
            f'<td>{info["value"]:.4f}</td>'
            f'<td>{info["count"]}</td></tr>\n'
        )

    # Build sample rows
    rows_html = []
    video_cache = {}  # scene -> frames (avoid re-reading same video)

    for idx, r in enumerate(results):
        sid = html.escape(r.get("sample_id", ""))
        dataset = r.get("dataset", "")
        scene = r.get("scene_name", "")
        qtype = r.get("question_type", "")
        question = r.get("question", "")
        predicted = r.get("predicted_answer", "")
        gt = r.get("ground_truth", "")
        model_text = r.get("final_answer_text", "")
        score_text, score_color = score_cell(r)

        # Correctness data attribute for filtering
        is_correct = "correct" if (r.get("accuracy") == 1.0 or (r.get("mra") is not None and r.get("mra", 0) >= 0.5)) else "wrong"

        # --- Input frames ---
        frames_html = ""
        video_key = f"{dataset}/{scene}"
        if video_key not in video_cache:
            video_path = os.path.join(data_dir, dataset, f"{scene}.mp4")
            if os.path.exists(video_path):
                try:
                    frames = sample_frames_from_video(video_path, n_frames=num_frames)
                    video_cache[video_key] = frames
                except Exception as e:
                    print(f"  [WARN] Failed to read {video_path}: {e}")
                    video_cache[video_key] = None
            else:
                video_cache[video_key] = None

        cached_frames = video_cache.get(video_key)
        if cached_frames:
            for fi, frame in enumerate(cached_frames):
                data_uri = img_to_base64(frame, thumb_size)
                frames_html += (
                    f'<div style="display:inline-block;text-align:center;margin:1px;">'
                    f'<div style="font-size:10px;color:#888;">F{fi}</div>'
                    f'<img src="{data_uri}" style="max-height:120px;border-radius:3px;">'
                    f'</div>'
                )
        else:
            frames_html = '<span style="color:gray;">[video not found]</span>'

        # --- Generated thinking images ---
        gen_html = ""
        saved_paths = r.get("saved_image_paths", [])
        # Also check generated_images_dir for {sample_id}_round_*.png
        if generated_images_dir and not saved_paths:
            pattern = os.path.join(generated_images_dir, f"{r.get('sample_id', '')}_round_*.png")
            saved_paths = sorted(glob.glob(pattern))

        for gp in saved_paths:
            # Resolve symlinks / alternative paths
            if not os.path.exists(gp):
                # Try relative to generated_images_dir
                alt = os.path.join(generated_images_dir or "", os.path.basename(gp))
                if os.path.exists(alt):
                    gp = alt
                else:
                    gen_html += f'<span style="color:gray;font-size:11px;">[missing: {os.path.basename(gp)}]</span> '
                    continue
            try:
                data_uri = file_to_base64(gp, thumb_size)
                label = os.path.basename(gp)
                gen_html += (
                    f'<div style="display:inline-block;text-align:center;margin:1px;">'
                    f'<div style="font-size:10px;color:#888;">{html.escape(label)}</div>'
                    f'<img src="{data_uri}" style="max-height:150px;border:2px solid #4CAF50;border-radius:3px;">'
                    f'</div>'
                )
            except Exception as e:
                gen_html += f'<span style="color:red;font-size:11px;">[error: {html.escape(str(e))}]</span> '

        if not gen_html:
            gen_html = '<span style="color:gray;font-size:11px;">none</span>'

        # --- Model output (truncate long text) ---
        display_text = model_text
        if len(display_text) > 1000:
            display_text = display_text[:1000] + "\n... [truncated]"

        rows_html.append(f"""
        <tr data-qtype="{html.escape(qtype)}" data-dataset="{html.escape(dataset)}" data-correct="{is_correct}">
            <td>{sid}</td>
            <td>{html.escape(dataset)}</td>
            <td>{html.escape(qtype)}</td>
            <td class="img-cell">{frames_html}</td>
            <td class="text-cell">{html.escape(question)}</td>
            <td class="text-cell model-output">{html.escape(display_text)}</td>
            <td class="img-cell">{gen_html}</td>
            <td><b>{html.escape(str(predicted) if predicted else "N/A")}</b></td>
            <td><b>{html.escape(str(gt))}</b></td>
            <td style="color:{score_color};font-weight:bold;">{score_text}</td>
        </tr>""")

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(results)} samples")

    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>VSI-Bench Results ({total} samples)</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #fff; }}
  h1 {{ font-size: 22px; }}
  .summary {{ margin-bottom: 20px; }}
  .summary table {{ border-collapse: collapse; margin: 8px 0; }}
  .summary th, .summary td {{ border: 1px solid #ddd; padding: 6px 12px; text-align: left; font-size: 13px; }}
  .summary th {{ background: #f0f0f0; }}
  .filters {{ margin: 16px 0; padding: 12px; background: #f9f9f9; border-radius: 6px; display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }}
  .filters label {{ font-size: 13px; font-weight: 600; }}
  .filters select {{ padding: 4px 8px; font-size: 13px; }}
  table.results {{ border-collapse: collapse; width: 100%; }}
  table.results th, table.results td {{ border: 1px solid #ccc; padding: 6px; vertical-align: top; text-align: left; }}
  table.results th {{ background: #f5f5f5; position: sticky; top: 0; z-index: 1; font-size: 12px; }}
  .text-cell {{ max-width: 300px; white-space: pre-wrap; word-break: break-word; font-size: 12px; }}
  .model-output {{ max-width: 400px; max-height: 200px; overflow-y: auto; font-size: 11px; color: #333; }}
  .img-cell {{ min-width: 200px; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  tr.hidden {{ display: none; }}
  .config {{ font-size: 12px; color: #666; margin: 4px 0; }}
</style>
</head>
<body>

<h1>VSI-Bench Results Preview</h1>
<div class="config">Model: <b>{html.escape(model_name)}</b> | Thinking: <b>{html.escape(thinking_mode)}</b> | Showing {len(results)} / {total} samples</div>

<div class="summary">
<h3>Summary</h3>
<table>
<tr><th>Metric</th><th>Value</th><th>Samples</th></tr>
<tr><td>Letter Accuracy</td><td><b>{acc_str}</b></td><td>{summary['total_letter']}</td></tr>
<tr><td>Numeric MRA</td><td><b>{mra_str}</b></td><td>{summary['total_numeric']}</td></tr>
</table>

<h4>Per Question Type</h4>
<table>
<tr><th>Question Type</th><th>Metric</th><th>Score</th><th>Count</th></tr>
{type_rows}
</table>
</div>

<div class="filters">
  <label>Question Type:</label>
  <select id="filter-qtype" onchange="applyFilters()">
    <option value="all">All</option>
  </select>
  <label>Dataset:</label>
  <select id="filter-dataset" onchange="applyFilters()">
    <option value="all">All</option>
  </select>
  <label>Result:</label>
  <select id="filter-correct" onchange="applyFilters()">
    <option value="all">All</option>
    <option value="correct">Correct</option>
    <option value="wrong">Wrong</option>
  </select>
  <span id="filter-count" style="font-size:13px;color:#666;margin-left:12px;"></span>
</div>

<table class="results">
<thead>
<tr>
  <th>ID</th><th>Dataset</th><th>Type</th><th>Input Frames</th><th>Question</th>
  <th>Model Output</th><th>Generated Images</th><th>Predicted</th><th>GT</th><th>Score</th>
</tr>
</thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>

<script>
// Populate filter dropdowns from data
(function() {{
  const rows = document.querySelectorAll('table.results tbody tr');
  const qtypes = new Set(), datasets = new Set();
  rows.forEach(r => {{
    qtypes.add(r.dataset.qtype);
    datasets.add(r.dataset.dataset);
  }});
  const qsel = document.getElementById('filter-qtype');
  [...qtypes].sort().forEach(q => {{
    const o = document.createElement('option'); o.value = q; o.textContent = q; qsel.appendChild(o);
  }});
  const dsel = document.getElementById('filter-dataset');
  [...datasets].sort().forEach(d => {{
    const o = document.createElement('option'); o.value = d; o.textContent = d; dsel.appendChild(o);
  }});
  applyFilters();
}})();

function applyFilters() {{
  const qtype = document.getElementById('filter-qtype').value;
  const dataset = document.getElementById('filter-dataset').value;
  const correct = document.getElementById('filter-correct').value;
  const rows = document.querySelectorAll('table.results tbody tr');
  let shown = 0;
  rows.forEach(r => {{
    let show = true;
    if (qtype !== 'all' && r.dataset.qtype !== qtype) show = false;
    if (dataset !== 'all' && r.dataset.dataset !== dataset) show = false;
    if (correct !== 'all' && r.dataset.correct !== correct) show = false;
    r.classList.toggle('hidden', !show);
    if (show) shown++;
  }});
  document.getElementById('filter-count').textContent = shown + ' / ' + rows.length + ' shown';
}}
</script>

</body>
</html>"""

    return page


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate HTML visualization of VSI-Bench results")
    parser.add_argument("--results_dir", required=True,
                        help="Directory with inference_results_shard*.json or inference_results_combined.json")
    parser.add_argument("--data_dir", default="/path/to/scratch/datasets/VSI-Bench",
                        help="Path to VSI-Bench videos")
    parser.add_argument("--output_html", default=None,
                        help="Output HTML path (default: tools/vsibench_results_preview.html)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Max samples to show (stratified by question_type)")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Frames to extract per video (must match inference)")
    parser.add_argument("--thumb_size", type=int, default=256,
                        help="Thumbnail max dimension")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_html is None:
        args.output_html = os.path.join(os.path.dirname(__file__), "vsibench_results_preview.html")

    thumb = (args.thumb_size, args.thumb_size)

    print(f"Loading results from {args.results_dir}...")
    results, config, _ = load_results(args.results_dir)
    print(f"Loaded {len(results)} results")

    # Compute summary on ALL results
    summary = compute_summary(results)

    # Sample subset for display
    if args.num_samples and args.num_samples < len(results):
        display_results = stratified_sample(results, args.num_samples, args.seed)
        print(f"Sampled {len(display_results)} for display (stratified)")
    else:
        display_results = results

    # Sort by question_type then sample_id for readability
    display_results.sort(key=lambda r: (r.get("question_type", ""), r.get("sample_id", "")))

    # Locate generated images dir
    gen_dir = os.path.join(args.results_dir, "generated_images")
    if not os.path.isdir(gen_dir):
        gen_dir = None

    print(f"Building HTML ({len(display_results)} samples)...")
    page = build_html(
        display_results, summary, config,
        data_dir=args.data_dir,
        num_frames=args.num_frames,
        thumb_size=thumb,
        generated_images_dir=gen_dir,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_html)), exist_ok=True)
    with open(args.output_html, "w") as f:
        f.write(page)

    print(f"Done! Wrote {args.output_html}")


if __name__ == "__main__":
    main()
