#!/usr/bin/env python3
"""Build teaser.html by inlining real images + real example questions.

Layout: top introduces the cross-view task and task families; bottom compares
four inference-time reasoning schemas for the same VLM.
"""

import argparse
import base64
import io
import json
import os
import re
from html import escape

from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(THIS_DIR, "teaser_template.html")
OUT_PATH = os.path.join(THIS_DIR, "teaser.html")

INFINIGEN_JSON_DIR = "/path/to/scratch/infinigen/spatial"
DEFAULT_V5_MAPPING = "/path/to/scratch/infinigen/scene_to_sample_id_v5_filtered.json"
DEFAULT_PAN_DIR = "/path/to/scratch/infinigen/rendered_panorama_v5"
DEFAULT_HERO_SAMPLE_ID = "relative_distance_000012"

MATTERPORT_VIEWS_ROOT = "/path/to/scratch/matterport_views"

CATEGORY_FILES = {
    "anchor":      f"{INFINIGEN_JSON_DIR}/dataset_anchor_questions_filtered_V5_normalized.json",
    "counting":    f"{INFINIGEN_JSON_DIR}/dataset_counting_questions_filtered_V5_normalized.json",
    "rel_dist":    f"{INFINIGEN_JSON_DIR}/dataset_relative_distance_questions_filtered_V5_normalized.json",
    "rel_dir":     f"{INFINIGEN_JSON_DIR}/dataset_spatial_questions_filtered_V5_normalized.json",
}

POINTMATCH_QA_GLOB = "rotation_qa_questions_point_match.json"
ROTATION_QA_GLOB = "rotation_qa_questions_two_image.json"


def encode_image(path: str, max_w: int | None = None, jpeg_quality: int = 88) -> str:
    with Image.open(path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        if max_w is not None and im.width > max_w:
            ratio = max_w / im.width
            im = im.resize((max_w, int(im.height * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def shorten(text: str, max_len: int = 95) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def first_question(json_path: str) -> str:
    with open(json_path) as f:
        data = json.load(f)
    for s in data:
        q = (s.get("question_both_views") or s.get("user_1_question") or "").strip()
        if q:
            return shorten(q)
    return "(no question)"


def _walk_matterport_qa(filename: str):
    """Yield parsed JSON from every <scene>/<vp>/<filename> under matterport root."""
    if not os.path.isdir(MATTERPORT_VIEWS_ROOT):
        return
    for scene in sorted(os.listdir(MATTERPORT_VIEWS_ROOT)):
        sdir = os.path.join(MATTERPORT_VIEWS_ROOT, scene)
        if not os.path.isdir(sdir):
            continue
        for vp in sorted(os.listdir(sdir)):
            qpath = os.path.join(sdir, vp, filename)
            if os.path.exists(qpath):
                try:
                    with open(qpath) as f:
                        yield json.load(f)
                except Exception:
                    continue


def find_pointmatch_question() -> str:
    for d in _walk_matterport_qa(POINTMATCH_QA_GLOB):
        for q in d.get("rotation_questions", []):
            text = q.get("question_text", "")
            first_sentence = re.split(r"(?<=[.?])\s+", text.strip())[0]
            if first_sentence:
                return shorten(first_sentence + " (which point corresponds?)")
    return shorten("Which point in image 2 corresponds to the same physical location as Q in image 1?")


def find_rotation_question() -> str:
    """Two-image rotation QA has the cleanest phrasing for a teaser figure."""
    for d in _walk_matterport_qa(ROTATION_QA_GLOB):
        for q in d.get("rotation_questions", []):
            text = q.get("question_text", "")
            # Drop the multi-line preamble; keep the actual asked sub-question.
            ask = re.search(r"(If I (?:was|am) facing[^?]*\?)", text)
            if ask:
                return shorten(ask.group(1))
            sentences = re.split(r"(?<=[.?])\s+", text.strip())
            for s in sentences:
                if s.endswith("?"):
                    return shorten(s)
    return shorten(
        "If I was facing the direction of image 1 and rotated to face image 2, which way did I turn?"
    )


def pick_hero_sample(rel_dist_json: str, v5_mapping_path: str, pan_dir: str,
                     preferred_sample_id: str):
    with open(v5_mapping_path) as f:
        v5_mapping = json.load(f)
    with open(rel_dist_json) as f:
        data = json.load(f)
    by_id = {s.get("sample_id"): s for s in data}
    candidates = [preferred_sample_id] + [k for k in by_id if k != preferred_sample_id]
    for sid in candidates:
        s = by_id.get(sid)
        if not s:
            continue
        scene_id = s.get("scene_id")
        if scene_id not in v5_mapping:
            continue
        pano_id = v5_mapping[scene_id]
        pano_path = os.path.join(pan_dir, scene_id, f"panorama_blender_limits_{pano_id}.png")
        img1 = s.get("user_1_image_local_path")
        img2 = s.get("user_2_image_local_path")
        options = s.get("options_user_1") or s.get("options", [])
        answer_idx = s.get("user_1_gt_answer_idx")
        question = (s.get("question_both_views") or s.get("user_1_question") or "").strip()
        if not (img1 and img2 and os.path.exists(img1) and os.path.exists(img2)):
            continue
        if not os.path.exists(pano_path):
            continue
        if not (options and answer_idx is not None and question and len(options) >= 4):
            continue
        return {
            "sample_id": sid, "scene_id": scene_id,
            "img1": img1, "img2": img2, "pano_gt": pano_path,
            "question": question, "options": options[:4], "answer_idx": answer_idx,
        }
    raise SystemExit("No usable hero sample found.")


def reorder_options_with_correct_first(options, answer_idx):
    if answer_idx == 0:
        return options[:4]
    correct = options[answer_idx]
    others = [o for i, o in enumerate(options) if i != answer_idx][:3]
    return [correct] + others


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5_mapping", default=DEFAULT_V5_MAPPING)
    ap.add_argument("--pan_dir", default=DEFAULT_PAN_DIR)
    # Kept for compatibility with older calls; the figure now uses the
    # panorama paired with the selected hero sample instead of a global pool.
    ap.add_argument("--generated_dir", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--sample_id", default=DEFAULT_HERO_SAMPLE_ID)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    print("[1/5] picking hero sample...")
    hero = pick_hero_sample(CATEGORY_FILES["rel_dist"], args.v5_mapping, args.pan_dir, args.sample_id)
    print(f"      -> {hero['sample_id']} (scene {hero['scene_id']})")

    print("[2/5] pulling per-task example questions...")
    ex = {
        "anchor":     first_question(CATEGORY_FILES["anchor"]),
        "counting":   first_question(CATEGORY_FILES["counting"]),
        "rel_dist":   first_question(CATEGORY_FILES["rel_dist"]),
        "rel_dir":    first_question(CATEGORY_FILES["rel_dir"]),
        "pointmatch": find_pointmatch_question(),
        "rotation":   find_rotation_question(),
    }
    for k, v in ex.items():
        print(f"      {k}: {v}")

    print("[3/5] using matching panorama for hero sample...")
    print(f"      -> {hero['pano_gt']}")

    print("[4/5] encoding task and panorama images...")
    cam0_b64 = encode_image(hero["img1"], max_w=720)
    cam1_b64 = encode_image(hero["img2"], max_w=720)
    pano_b64 = encode_image(hero["pano_gt"], max_w=900)

    print("[5/5] filling template...")
    with open(TEMPLATE_PATH) as f:
        html = f.read()

    options = reorder_options_with_correct_first(hero["options"], hero["answer_idx"])
    repl = {
        "__CAM0_B64__":       cam0_b64,
        "__CAM1_B64__":       cam1_b64,
        "__PANO_B64__":       pano_b64,
        "__QUESTION__":       escape(hero["question"]),
        "__OPT_A__":          escape(options[0]),
        "__OPT_B__":          escape(options[1]),
        "__OPT_C__":          escape(options[2]),
        "__OPT_D__":          escape(options[3]),
        "__EX_ANCHOR__":      escape(ex["anchor"]),
        "__EX_COUNTING__":    escape(ex["counting"]),
        "__EX_RELDIST__":     escape(ex["rel_dist"]),
        "__EX_RELDIR__":      escape(ex["rel_dir"]),
        "__EX_POINTMATCH__":  escape(ex["pointmatch"]),
        "__EX_ROTATION__":    escape(ex["rotation"]),
    }
    for k, v in repl.items():
        html = html.replace(k, v)

    with open(args.out, "w") as f:
        f.write(html)
    print(f"\nWrote {args.out} ({os.path.getsize(args.out) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
