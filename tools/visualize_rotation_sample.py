#!/usr/bin/env python3
"""
Visualize rotation training samples.

Picks one sample from rotation_qa_questions.json and renders:
  - Row 1: image_1, image_2, image_3, image_4  (wall views: top/right/bottom/left)
  - Row 2: bridge_bev  (BEV thinking image)
  - Text:  question, options, correct answer, sample_id

Usage:
    python tools/visualize_rotation_sample.py
    python tools/visualize_rotation_sample.py --rotation_dir /path/to/novel_qa_rotation --index 0 --output out.png
"""

import argparse
import json
import os
import random
import textwrap

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


ROTATION_DIR = "/path/to/scratch/VisualCoT/novel_qa_rotation"


def find_all_rotation_jsons(rotation_dir: str) -> list:
    found = []
    for dirpath, dirnames, filenames in os.walk(rotation_dir):
        dirnames.sort()
        if "rotation_qa_questions.json" in filenames:
            found.append(os.path.join(dirpath, "rotation_qa_questions.json"))
    return sorted(found)


def load_questions_from_json(json_path: str) -> list:
    """Return list of (novel_qa_dir, question_dict)."""
    novel_qa_dir = os.path.dirname(json_path)
    with open(json_path) as f:
        data = json.load(f)
    return [(novel_qa_dir, q) for q in data.get("rotation_questions", [])]


def visualize_sample(novel_qa_dir: str, question: dict, output_path: str):
    images_dict = question.get("images", {})
    img_keys = ["image_1", "image_2", "image_3", "image_4"]
    wall_labels = ["image_1\n(wall top)", "image_2\n(wall right)", "image_3\n(wall bottom)", "image_4\n(wall left)"]
    bev_key = "bridge_bev"

    imgs = []
    for k in img_keys:
        p = os.path.join(novel_qa_dir, images_dict.get(k, ""))
        imgs.append(Image.open(p).convert("RGB"))

    bev_path = os.path.join(novel_qa_dir, images_dict.get(bev_key, ""))
    bev_img = Image.open(bev_path).convert("RGB")

    question_type = question.get("question_type", "")
    question_text = question.get("question", "")
    options = question.get("options", [])
    correct_index = question.get("correct_index")
    correct_answer = question.get("correct_answer", "")
    answer_letter = chr(65 + correct_index) if correct_index is not None else "?"

    # Build annotation text
    if options:
        options_str = "  ".join(f"{chr(65+i)}) {o}" for i, o in enumerate(options))
    else:
        options_str = ""

    wrapped_q = textwrap.fill(question_text, width=55)
    if options:
        options_str = "\n".join(f"  {chr(65+i)}) {o}" for i, o in enumerate(options))
    else:
        options_str = ""
    annotation = (
        f"[{question_type}]\n\n"
        f"Q: {wrapped_q}\n"
        + (f"\n{options_str}\n" if options_str else "")
        + f"\nAnswer: {answer_letter}) {correct_answer}"
    )

    # --- Layout: row 0 = 4 perspective images; row 1 = BEV (left) + question text (right) ---
    fig = plt.figure(figsize=(22, 10))
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[2.5, 3.5],
        width_ratios=[1, 1],
        hspace=0.25,
        wspace=0.12,
    )

    # Row 0: 4 perspective images — use a nested gridspec inside gs[0, :]
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0, :], wspace=0.06)
    for i, (img, label) in enumerate(zip(imgs, wall_labels)):
        ax = fig.add_subplot(gs_top[0, i])
        ax.imshow(img)
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    # Row 1, left: BEV image (large)
    ax_bev = fig.add_subplot(gs[1, 0])
    ax_bev.imshow(bev_img)
    ax_bev.set_title("bridge_bev  (BEV thinking image)", fontsize=11)
    ax_bev.axis("off")

    # Row 1, right: annotation text
    ax_txt = fig.add_subplot(gs[1, 1])
    ax_txt.axis("off")
    ax_txt.text(
        0.0, 1.0, annotation,
        transform=ax_txt.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        fontfamily="monospace",
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#aaa"),
    )

    path_parts = novel_qa_dir.rstrip("/").split("/")
    scene_id = path_parts[-2] if len(path_parts) >= 2 else "?"
    room_type = path_parts[-3] if len(path_parts) >= 3 else "?"
    fig.suptitle(f"Rotation Sample  |  {room_type} / {scene_id}", fontsize=13, fontweight="bold")

    plt.savefig(output_path, bbox_inches="tight", dpi=120)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotation_dir", default=ROTATION_DIR)
    parser.add_argument("--index", type=int, default=None,
                        help="Global question index across all JSONs. Random if omitted.")
    parser.add_argument("--output", default="rotation_sample_vis.png")
    parser.add_argument("--question_type", default=None,
                        help="Filter by question_type (e.g. rotation_direction_mcq, rotation_proximity_yesno)")
    args = parser.parse_args()

    json_files = find_all_rotation_jsons(args.rotation_dir)
    if not json_files:
        print(f"No rotation_qa_questions.json found under {args.rotation_dir}")
        return

    all_pairs = []
    for jf in json_files:
        all_pairs.extend(load_questions_from_json(jf))

    if args.question_type:
        all_pairs = [(d, q) for d, q in all_pairs if q.get("question_type") == args.question_type]

    if not all_pairs:
        print("No matching samples found.")
        return

    print(f"Total rotation questions available: {len(all_pairs)}")

    if args.index is not None:
        idx = args.index % len(all_pairs)
    else:
        idx = random.randint(0, len(all_pairs) - 1)

    novel_qa_dir, question = all_pairs[idx]
    print(f"Visualizing index {idx}: {novel_qa_dir}  [{question.get('question_type')}]")
    visualize_sample(novel_qa_dir, question, args.output)


if __name__ == "__main__":
    main()
