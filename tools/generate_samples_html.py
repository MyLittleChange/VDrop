"""Sample 100 entries from metadata.jsonl and generate a self-contained HTML page with embedded images."""

import base64
import html
import json
import os
import random
import re
from io import BytesIO

from PIL import Image

METADATA_PATH = "/path/to/scratch/datasets/ViewFusion/SFT_data/metadata.jsonl"
IMAGE_ROOT = "/path/to/scratch/datasets/ViewFusion"
OUTPUT_HTML = "/path/to/ThinkMorph-BAGEL-release/tools/samples_preview.html"
NUM_SAMPLES = 100
SEED = 42
THUMB_SIZE = (256, 256)


def split_response(response: str):
    """Split response into spatial_thinking, thinking, and answer parts."""
    spatial = ""
    thinking = ""
    answer = ""

    m = re.search(r"<spatial_thinking>(.*?)</spatial_thinking>", response, re.DOTALL)
    if m:
        spatial = m.group(1).strip()

    m = re.search(r"<thinking>(.*?)</thinking>", response, re.DOTALL)
    if m:
        thinking = m.group(1).strip()

    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if m:
        answer = m.group(1).strip()

    return spatial, thinking, answer


def img_to_base64(path):
    """Load image, resize to thumbnail, return base64 data URI."""
    img = Image.open(path).convert("RGB")
    img.thumbnail(THUMB_SIZE)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def main():
    with open(METADATA_PATH, "r") as f:
        lines = f.readlines()

    random.seed(SEED)
    sampled_lines = random.sample(lines, min(NUM_SAMPLES, len(lines)))
    parsed_entries = [json.loads(line) for line in sampled_lines]

    rows_html = []
    for i, entry in enumerate(parsed_entries):
        images_html = ""
        for img_path in entry.get("images", []):
            full_path = os.path.join(IMAGE_ROOT, img_path)
            if os.path.exists(full_path):
                try:
                    data_uri = img_to_base64(full_path)
                    images_html += f'<img src="{data_uri}" style="max-height:200px;margin:2px;">'
                except Exception as e:
                    images_html += f'<span style="color:red;">[error: {html.escape(str(e))}]</span>'
            else:
                images_html += '<span style="color:gray;">[missing]</span>'

        spatial, thinking, answer_parsed = split_response(entry.get("response", ""))

        rows_html.append(f"""
        <tr>
            <td>{html.escape(str(entry.get("id", "")))}</td>
            <td class="text-cell">{html.escape(entry.get("question", ""))}</td>
            <td class="img-cell">{images_html}</td>
            <td>{html.escape(entry.get("answer", ""))}</td>
            <td class="text-cell">{html.escape(spatial)}</td>
            <td class="text-cell">{html.escape(thinking)}</td>
        </tr>""")

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(parsed_entries)} entries")

    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ViewFusion Samples ({NUM_SAMPLES})</title>
<style>
  body {{ font-family: sans-serif; margin: 20px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 8px; vertical-align: top; text-align: left; }}
  th {{ background: #f5f5f5; position: sticky; top: 0; }}
  .text-cell {{ max-width: 400px; white-space: pre-wrap; word-break: break-word; font-size: 13px; }}
  .img-cell {{ min-width: 200px; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  img {{ border-radius: 4px; }}
</style>
</head>
<body>
<h1>ViewFusion Sample Preview ({NUM_SAMPLES} entries)</h1>
<table>
<thead>
<tr><th>ID</th><th>Question</th><th>Images</th><th>Answer</th><th>Spatial Thinking</th><th>Thinking</th></tr>
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
