import base64
import json
import random
import re
import sys
from pathlib import Path

SEED = 42
SAMPLE_N = 100

DIR1 = Path("/path/to/scratch/infinigen/training_data_mix_all_balance/text_thinking")
DIR1_FILES = ["anchor.jsonl", "counting.jsonl", "perspective_taking.jsonl", "relative_distance.jsonl", "spatial.jsonl"]
DIR2_FILE = Path("/path/to/scratch/infinigen/training_data_point_matching/text_thinking/point_matching.jsonl")
DIR3_FILE = Path("/path/to/scratch/matterport/training_data_point_matching_matterport/text_thinking/matterport_point_matching.jsonl")
DIR4_FILE = Path("/path/to/scratch/infinigen/training_data_matterport_rotation/text_thinking/matterport_rotation.jsonl")

OUTPUT = Path(__file__).parent / "annotation_viewer.html"


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


SCRATCH_ROOT = Path("/network/scratch")


def resolve_image_path(path_str):
    p = Path(path_str)
    if p.is_absolute():
        return p
    # Paths in the JSONL are relative to /network/scratch/
    return SCRATCH_ROOT / p


MAX_IMG_WIDTH = 640


def encode_image(path_str):
    p = resolve_image_path(path_str)
    if not p.exists():
        return None
    try:
        from PIL import Image
        import io
        img = Image.open(p).convert("RGB")
        if img.width > MAX_IMG_WIDTH:
            ratio = MAX_IMG_WIDTH / img.width
            img = img.resize((MAX_IMG_WIDTH, int(img.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        data = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{data}"
    except Exception:
        # Fallback: embed raw bytes without resize
        suffix = p.suffix.lower()
        mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        with open(p, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:{mime};base64,{data}"


def parse_gpt_response(text):
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    think = think_match.group(1).strip() if think_match else ""
    answer = answer_match.group(1).strip() if answer_match else text.strip()
    return think, answer


def infer_task_type(record, source_file):
    # Try id prefix first
    rid = record.get("id", "")
    for task in ("anchor", "counting", "perspective_taking", "relative_distance", "spatial", "point_matching", "pointmatch"):
        if rid.startswith(task):
            return task.replace("_", " ").title()
    # Fall back to filename
    return Path(source_file).stem.replace("_", " ").title()


def build_question_html(human_text, img_tags):
    """Replace <image> placeholders with actual <img> tags."""
    parts = human_text.split("<image>")
    result = []
    for i, part in enumerate(parts):
        if part:
            result.append(f'<span class="q-text">{part}</span>')
        if i < len(parts) - 1 and i < len(img_tags):
            result.append(img_tags[i])
    return "".join(result)


def sample_records(files_or_file, n, label):
    all_records = []
    if isinstance(files_or_file, list):
        for fname in files_or_file:
            p = DIR1 / fname
            if not p.exists():
                print(f"  Missing: {p}", file=sys.stderr)
                continue
            recs = load_jsonl(p)
            for r in recs:
                r["_source_file"] = fname
            all_records.extend(recs)
    else:
        recs = load_jsonl(files_or_file)
        for r in recs:
            r["_source_file"] = files_or_file.name
        all_records.extend(recs)

    print(f"{label}: {len(all_records)} total records")
    sampled = random.sample(all_records, min(n, len(all_records)))
    print(f"  Sampled: {len(sampled)}")
    return sampled


def render_card(record, idx, section_id):
    human_text = ""
    gpt_text = ""
    for turn in record.get("conversations", []):
        if turn["from"] == "human":
            human_text = turn["value"]
        elif turn["from"] == "gpt":
            gpt_text = turn["value"]

    image_paths = record.get("image", [])
    img_tags = []
    skipped_imgs = 0
    for ip in image_paths:
        encoded = encode_image(ip)
        if encoded:
            img_tags.append(f'<img src="{encoded}" style="max-width:380px;max-height:320px;border-radius:4px;border:1px solid #ddd;">')
        else:
            img_tags.append('<span class="missing-img">[image not found]</span>')
            skipped_imgs += 1

    task_type = infer_task_type(record, record.get("_source_file", ""))
    record_id = record.get("id", f"record_{idx}")
    think, answer = parse_gpt_response(gpt_text)

    imgs_html = '<div class="imgs-row">' + "".join(img_tags) + "</div>" if img_tags else ""
    question_html = build_question_html(human_text, img_tags)

    think_html = ""
    if think:
        escaped_think = think.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        think_html = f"""<details class="think-block">
  <summary>Thinking</summary>
  <pre class="think-content">{escaped_think}</pre>
</details>"""

    answer_html = f'<div class="answer-block"><span class="answer-label">Answer:</span> <span class="answer-value">{answer}</span></div>'

    return f"""<div class="card" id="{section_id}-{idx}">
  <div class="card-header">
    <span class="card-num">#{idx + 1}</span>
    <span class="task-tag">{task_type}</span>
    <span class="record-id">{record_id}</span>
  </div>
  <div class="question-block">{question_html}</div>
  {think_html}
  {answer_html}
</div>"""


def render_section(records, section_id, title):
    cards = "".join(render_card(r, i, section_id) for i, r in enumerate(records))
    return f"""<div id="{section_id}" class="section">
  <h2>{title}</h2>
  {cards}
</div>"""


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; color: #222; }
.tab-bar { position: sticky; top: 0; background: #fff; border-bottom: 2px solid #e0e0e0; display: flex; gap: 0; z-index: 100; }
.tab-btn { padding: 14px 28px; cursor: pointer; font-size: 15px; font-weight: 500; border: none; background: none; color: #666; border-bottom: 3px solid transparent; margin-bottom: -2px; transition: all .15s; }
.tab-btn.active { color: #2563eb; border-bottom-color: #2563eb; }
.tab-btn:hover:not(.active) { background: #f0f4ff; color: #444; }
.section { display: none; max-width: 900px; margin: 0 auto; padding: 24px 16px 60px; }
.section.active { display: block; }
.card { background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 28px; padding: 20px 24px; }
.card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
.card-num { font-size: 18px; font-weight: 700; color: #888; min-width: 36px; }
.task-tag { background: #e0e7ff; color: #3730a3; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }
.record-id { font-size: 11px; color: #aaa; font-family: monospace; margin-left: auto; }
.imgs-row { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 14px; }
.missing-img { color: #f87171; font-size: 12px; padding: 8px; background: #fee2e2; border-radius: 4px; }
.question-block { font-size: 14px; line-height: 1.7; margin-bottom: 14px; }
.question-block img { display: block; margin: 8px 0; }
.q-text { white-space: pre-wrap; }
.think-block { margin-bottom: 12px; }
.think-block summary { cursor: pointer; font-size: 13px; font-weight: 600; color: #6b7280; padding: 6px 0; user-select: none; }
.think-block summary:hover { color: #374151; }
.think-content { font-size: 12px; color: #374151; background: #f9fafb; border-left: 3px solid #d1d5db; padding: 10px 14px; margin-top: 6px; white-space: pre-wrap; word-break: break-word; border-radius: 0 4px 4px 0; max-height: 400px; overflow-y: auto; }
.answer-block { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 10px 14px; font-size: 14px; }
.answer-label { font-weight: 700; color: #166534; }
.answer-value { color: #15803d; font-weight: 600; }
h2 { font-size: 20px; font-weight: 700; color: #1e3a5f; margin-bottom: 20px; }
"""

JS = """
function showTab(id) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelector('[data-tab="' + id + '"]').classList.add('active');
}
document.addEventListener('DOMContentLoaded', () => showTab('mix'));
"""


def main():
    random.seed(SEED)
    print(f"Random seed: {SEED}")

    mix_records = sample_records(DIR1_FILES, SAMPLE_N, "Mix All Balance")
    pm_records = sample_records(DIR2_FILE, SAMPLE_N, "Point Matching (Infinigen)")
    matterport_pm_records = sample_records(DIR3_FILE, SAMPLE_N, "Point Matching (Matterport)")
    matterport_rot_records = sample_records(DIR4_FILE, SAMPLE_N, "Matterport Rotation")

    mix_html = render_section(mix_records, "mix", f"Mix All Balance — {len(mix_records)} samples")
    pm_html = render_section(pm_records, "pm", f"Point Matching Infinigen — {len(pm_records)} samples")
    matterport_pm_html = render_section(matterport_pm_records, "mpm", f"Point Matching Matterport — {len(matterport_pm_records)} samples")
    matterport_rot_html = render_section(matterport_rot_records, "mrot", f"Matterport Rotation — {len(matterport_rot_records)} samples")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Annotation Quality Viewer (seed={SEED})</title>
<style>{CSS}</style>
</head>
<body>
<div class="tab-bar">
  <button class="tab-btn" data-tab="mix" onclick="showTab('mix')">Mix All Balance ({len(mix_records)})</button>
  <button class="tab-btn" data-tab="pm" onclick="showTab('pm')">PM Infinigen ({len(pm_records)})</button>
  <button class="tab-btn" data-tab="mpm" onclick="showTab('mpm')">PM Matterport ({len(matterport_pm_records)})</button>
  <button class="tab-btn" data-tab="mrot" onclick="showTab('mrot')">Matterport Rotation ({len(matterport_rot_records)})</button>
</div>
{mix_html}
{pm_html}
{matterport_pm_html}
{matterport_rot_html}
<script>{JS}</script>
</body>
</html>"""

    OUTPUT.write_text(html)
    print(f"\nOutput written to: {OUTPUT}")
    print(f"File size: {OUTPUT.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
