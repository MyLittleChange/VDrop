"""Sample 100 entries from metadata.jsonl and upload to W&B as a table with images."""

import json
import os
import random
import re

from PIL import Image

import wandb

METADATA_PATH = "/path/to/scratch/datasets/ViewFusion/SFT_data/metadata.jsonl"
IMAGE_ROOT = "/path/to/scratch/datasets/ViewFusion"
NUM_SAMPLES = 100
SEED = 42
THUMB_SIZE = (256, 256)

PROJECT = "thinkmorph-orbit-annotations"


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


def load_thumbnail(path):
    """Load image and resize to thumbnail for W&B upload."""
    img = Image.open(path).convert("RGB")
    img.thumbnail(THUMB_SIZE)
    return img


def main():
    # Load all lines
    with open(METADATA_PATH, "r") as f:
        lines = f.readlines()

    # Sample
    random.seed(SEED)
    sampled_lines = random.sample(lines, min(NUM_SAMPLES, len(lines)))
    parsed_entries = [json.loads(line) for line in sampled_lines]

    # Use a single "images" column with a list of images to avoid None-padding issues
    columns = ["id", "question", "images", "answer", "spatial_thinking", "thinking"]

    # Init wandb
    print("Initializing W&B run...")
    run = wandb.init(project=PROJECT, name="ViewFusion-sample-100-preview")

    # Build table row by row
    table = wandb.Table(columns=columns)
    for i, entry in enumerate(parsed_entries):
        wb_images = []
        for img_path in entry.get("images", []):
            full_path = os.path.join(IMAGE_ROOT, img_path)
            if os.path.exists(full_path):
                try:
                    wb_images.append(wandb.Image(load_thumbnail(full_path)))
                except Exception as e:
                    print(f"Warning: Could not load image {full_path}: {e}")
            else:
                print(f"Warning: Image not found: {full_path}")

        spatial, thinking, answer_parsed = split_response(entry.get("response", ""))

        table.add_data(
            entry.get("id", ""),
            entry.get("question", ""),
            wb_images,
            entry.get("answer", ""),
            spatial,
            thinking,
        )
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(parsed_entries)} entries")

    print(f"Uploading table with {len(table.data)} rows to W&B...")
    run.log({"samples": table})

    wandb.finish()
    print(f"Done! View results at: {run.url}")


if __name__ == "__main__":
    main()
