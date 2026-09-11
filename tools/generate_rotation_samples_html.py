"""Sample 100 rotation QA entries and generate a self-contained HTML page with embedded images."""

import base64
import html
import json
import os
import random
from io import BytesIO

from PIL import Image

ROTATION_QA_DIR = "/path/to/scratch/VisualCoT/novel_qa_rotation_spatial"
OUTPUT_HTML = "/path/to/ThinkMorph-BAGEL-release/tools/rotation_samples_preview.html"
NUM_SAMPLES = 100
SEED = 42
THUMB_SIZE = (256, 256)


def load_all_rotation_samples(rotation_qa_dir: str) -> list:
    """Walk rotation QA dir and collect all questions from all scenes."""
    samples = []
    for room_part in sorted(os.listdir(rotation_qa_dir)):
        room_dir = os.path.join(rotation_qa_dir, room_part)
        if not os.path.isdir(room_dir):
            continue
        for scene_id in sorted(os.listdir(room_dir)):
            qa_file = os.path.join(room_dir, scene_id, "novel_qa", "rotation_qa_questions.json")
            if not os.path.exists(qa_file):
                continue
            try:
                with open(qa_file, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  [WARNING] Failed to load {qa_file}: {e}")
                continue

            novel_qa_dir = os.path.join(room_dir, scene_id, "novel_qa")
            for i, q in enumerate(data.get("rotation_questions", [])):
                correct_index = q.get("correct_index")
                question_type = q.get("question_type", "")
                if question_type == "rotation_direction_mcq":
                    correct_answer = chr(65 + correct_index) if correct_index is not None else q.get("correct_answer", "")
                else:
                    correct_answer = q.get("correct_answer", "")

                options = q.get("options", [])
                question = q.get("question", "")
                if options:
                    options_str = "\n".join([f"{chr(65 + j)}) {opt}" for j, opt in enumerate(options)])
                    full_question = f"{question}\n\n{options_str}"
                else:
                    full_question = question

                samples.append({
                    "sample_id": f"rotation_{scene_id}_{i}",
                    "question_type": question_type,
                    "novel_qa_dir": novel_qa_dir,
                    "question": full_question,
                    "correct_answer": correct_answer,
                    "images": q.get("images", {}),
                })
    return samples


def img_to_base64(path):
    """Load image, resize to thumbnail, return base64 data URI."""
    img = Image.open(path).convert("RGB")
    img.thumbnail(THUMB_SIZE)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def main():
    print(f"Loading rotation samples from {ROTATION_QA_DIR}...")
    all_samples = load_all_rotation_samples(ROTATION_QA_DIR)
    print(f"Found {len(all_samples)} total rotation samples.")

    random.seed(SEED)
    sampled = random.sample(all_samples, min(NUM_SAMPLES, len(all_samples)))

    img_keys = ["image_1", "image_2", "image_3", "image_4", "bridge_panorama"]
    img_labels = ["Wall 1", "Wall 2", "Wall 3", "Wall 4", "Panorama"]

    rows_html = []
    for i, sample in enumerate(sampled):
        images_html = ""
        for key, label in zip(img_keys, img_labels):
            filename = sample["images"].get(key)
            if not filename:
                continue
            path = os.path.join(sample["novel_qa_dir"], filename)
            if os.path.exists(path):
                try:
                    data_uri = img_to_base64(path)
                    images_html += (
                        f'<div style="display:inline-block;text-align:center;margin:2px;">'
                        f'<div style="font-size:11px;color:#666;">{label}</div>'
                        f'<img src="{data_uri}" style="max-height:180px;border-radius:4px;">'
                        f'</div>'
                    )
                except Exception as e:
                    images_html += f'<span style="color:red;">[error: {html.escape(str(e))}]</span>'
            else:
                images_html += f'<span style="color:gray;">[missing: {label}]</span>'

        rows_html.append(f"""
        <tr>
            <td>{html.escape(sample["sample_id"])}</td>
            <td>{html.escape(sample["question_type"])}</td>
            <td class="text-cell">{html.escape(sample["question"])}</td>
            <td class="img-cell">{images_html}</td>
            <td><b>{html.escape(sample["correct_answer"])}</b></td>
        </tr>""")

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(sampled)} entries")

    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Rotation Samples ({NUM_SAMPLES})</title>
<style>
  body {{ font-family: sans-serif; margin: 20px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 8px; vertical-align: top; text-align: left; }}
  th {{ background: #f5f5f5; position: sticky; top: 0; z-index: 1; }}
  .text-cell {{ max-width: 350px; white-space: pre-wrap; word-break: break-word; font-size: 13px; }}
  .img-cell {{ min-width: 300px; }}
  tr:nth-child(even) {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>Rotation QA Sample Preview ({NUM_SAMPLES} entries)</h1>
<table>
<thead>
<tr><th>ID</th><th>Type</th><th>Question</th><th>Images</th><th>Answer</th></tr>
</thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>
</body>
</html>"""

    with open(OUTPUT_HTML, "w") as f:
        f.write(page)

    print(f"Done! Wrote {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
