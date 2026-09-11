"""Sample 100 entries from no_thinking.jsonl and upload to W&B as a table with images."""

import json
import os
import random
import re

from PIL import Image

import wandb

JSONL_PATH = "/path/to/scratch/infinigen/training_data_mix_all/no_thinking/no_thinking.jsonl"
IMAGE_ROOT = "/network/scratch"
NUM_SAMPLES = 100
SEED = 42
THUMB_SIZE = (256, 256)

PROJECT = "thinkmorph-orbit-annotations"


def load_thumbnail(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail(THUMB_SIZE)
    return img


def main():
    with open(JSONL_PATH, "r") as f:
        lines = f.readlines()

    random.seed(SEED)
    sampled_lines = random.sample(lines, min(NUM_SAMPLES, len(lines)))
    entries = [json.loads(line) for line in sampled_lines]

    columns = ["id", "question", "answer", "images"]

    print("Initializing W&B run...")
    run = wandb.init(project=PROJECT, name="no_thinking-sample-100-preview")

    table = wandb.Table(columns=columns)
    for i, entry in enumerate(entries):
        convs = entry.get("conversations", [])
        question = next((c["value"] for c in convs if c["from"] == "human"), "")
        answer = next((c["value"] for c in convs if c["from"] == "gpt"), "")

        wb_images = []
        for img_path in entry.get("image", []):
            full_path = os.path.join(IMAGE_ROOT, img_path)
            if os.path.exists(full_path):
                try:
                    wb_images.append(wandb.Image(load_thumbnail(full_path)))
                except Exception as e:
                    print(f"Warning: Could not load image {full_path}: {e}")
            else:
                print(f"Warning: Image not found: {full_path}")

        table.add_data(
            entry.get("id", ""),
            question,
            answer,
            wb_images,
        )
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(entries)} entries")

    print(f"Uploading table with {len(table.data)} rows to W&B...")
    run.log({"samples": table})
    wandb.finish()
    print(f"Done! View at: {run.url}")


if __name__ == "__main__":
    main()
