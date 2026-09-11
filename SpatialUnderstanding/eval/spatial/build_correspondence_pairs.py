#!/usr/bin/env python3
"""
T_cor: render correspondence-overlay versions of V1 + V2 for the COSMIC
test split (anchor + counting subtasks only — relative_distance and
relative_direction don't benefit from object correspondence per design §6).

Approach: read `visible_objects.json` per scene, take the intersection of
object names visible in both camera_0_0 and camera_1_0 (= co-visible
objects), pick up to 3 of them deterministically per sample (preferring
objects mentioned in the question/options), and draw each object's
2D bbox + label on V1 and V2 in a consistent color across views.

bbox_2d in visible_objects.json is normalized to [0, 1] (top-left origin)
in both axes — values can go negative or > 1 when an object extends off
the visible frame; we clip to the image bounds.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CMAP = [
    (255, 56, 56),
    (56, 200, 60),
    (60, 130, 255),
    (255, 195, 0),
    (200, 60, 230),
    (0, 200, 220),
]

TEST_JSONS = {
    "anchor": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
    "counting": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
    "relative_distance": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
    "relative_direction": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
}
SCENE_ROOT = Path("/path/to/scratch/infinigen/infinigen_debang/infinigen/v00_filtered")


def find_scene_dir(sample):
    """Walk up from the image path until we hit a dir containing visible_objects.json."""
    img_path = Path(sample["user_1_image_local_path"])
    for ancestor in img_path.parents:
        if (ancestor / "visible_objects.json").exists():
            return ancestor
    return None


def normalize_object_name(name):
    return re.sub(r"\s+\d+$", "", name).lower()


def pick_objects(covisible, question_text, options_text, sample_id, k=3):
    """Pick up to k objects, preferring ones mentioned in question/options."""
    qstr = (question_text + " " + options_text).lower()
    mentioned, others = [], []
    for obj in covisible:
        base = normalize_object_name(obj)
        if base and base in qstr:
            mentioned.append(obj)
        else:
            others.append(obj)
    rng_seed = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16)
    mentioned.sort()
    others.sort()
    picks = mentioned[:k]
    if len(picks) < k:
        # Deterministic pick from others
        idx = rng_seed % max(len(others), 1)
        for o in others[idx:] + others[:idx]:
            if o not in picks:
                picks.append(o)
            if len(picks) >= k:
                break
    return picks[:k]


def draw_overlay(img_path, picks, vobj_for_cam, color_for_obj, out_path):
    """Open V_i, draw per-object bbox+label, save to out_path."""
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(16, H // 40))
    except OSError:
        font = ImageFont.load_default()

    for obj in picks:
        if obj not in vobj_for_cam:
            continue
        bb = vobj_for_cam[obj].get("bbox_2d")
        if not bb or len(bb) != 4:
            continue
        x_min, y_min, x_max, y_max = bb
        # Normalized [0, 1] (top-left origin) → pixel coords; clip to image
        px_min = max(0, int(x_min * W))
        py_min = max(0, int(y_min * H))
        px_max = min(W - 1, int(x_max * W))
        py_max = min(H - 1, int(y_max * H))
        if px_max <= px_min or py_max <= py_min:
            continue
        # Skip if entirely off-screen
        if x_max < 0 or y_max < 0 or x_min > 1 or y_min > 1:
            continue
        color = color_for_obj[obj]
        line_w = max(3, H // 200)
        draw.rectangle([(px_min, py_min), (px_max, py_max)], outline=color, width=line_w)
        # Label background + text
        label = obj
        tx = px_min + 4
        ty = max(0, py_min - line_w - font.size - 4)
        try:
            tw = draw.textlength(label, font=font)
        except AttributeError:
            tw = len(label) * (font.size // 2)
        draw.rectangle([(tx - 2, ty - 2), (tx + tw + 4, ty + font.size + 4)], fill=color)
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def process_subtask(subtask, json_path, output_root):
    samples = json.load(open(json_path))
    n_ok = n_skipped = 0
    skip_reasons = {}
    out_dir = output_root / f"correspondence_approved_mcqs_{subtask}"
    for sample in samples:
        sid = sample["sample_id"]
        scene_id = sample["scene_id"]
        scene_dir = find_scene_dir(sample)
        if scene_dir is None or not scene_dir.exists():
            n_skipped += 1
            skip_reasons[sid] = "scene_dir_missing"
            continue
        vobj_path = scene_dir / "visible_objects.json"
        if not vobj_path.exists():
            n_skipped += 1
            skip_reasons[sid] = "visible_objects_missing"
            continue
        vobj = json.load(open(vobj_path))
        cam0 = vobj.get("camera_0_0", {})
        cam1 = vobj.get("camera_1_0", {})
        covisible = sorted(set(cam0.keys()) & set(cam1.keys()))
        if not covisible:
            n_skipped += 1
            skip_reasons[sid] = "no_covisible_objects"
            continue

        # Pick up to 3 objects, prefer those mentioned in the question
        asking_to = sample.get("asking_to", "agent_2")
        if asking_to == "agent_1":
            q = sample.get("user_1_question", "") or ""
            opts = sample.get("options_user_1") or []
        else:
            q = sample.get("user_2_question", "") or ""
            opts = sample.get("options_user_2") or []
        opts_str = " ".join(opts) if isinstance(opts, list) else (opts or "")
        picks = pick_objects(covisible, q, opts_str, sid, k=3)
        if not picks:
            n_skipped += 1
            skip_reasons[sid] = "no_picks"
            continue

        color_for_obj = {obj: CMAP[i % len(CMAP)] for i, obj in enumerate(picks)}

        out_v1 = out_dir / scene_id / f"cor_{sid}_v1.png"
        out_v2 = out_dir / scene_id / f"cor_{sid}_v2.png"
        draw_overlay(sample["user_1_image_local_path"], picks, cam0, color_for_obj, out_v1)
        draw_overlay(sample["user_2_image_local_path"], picks, cam1, color_for_obj, out_v2)
        n_ok += 1

    summary = {"subtask": subtask, "n_ok": n_ok, "n_skipped": n_skipped, "skip_reasons": skip_reasons}
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  {subtask}: ok={n_ok}, skipped={n_skipped}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root",
        default="/path/to/scratch/VisualCoT/infinigen",
    )
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for subtask, jp in TEST_JSONS.items():
        process_subtask(subtask, jp, output_root)


if __name__ == "__main__":
    main()
