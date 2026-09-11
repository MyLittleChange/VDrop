"""Sample 100 rotation QA entries and upload to W&B as a table with images."""

import json
import os
import random

from PIL import Image

import wandb

ROTATION_QA_DIR = "/path/to/scratch/VisualCoT/novel_qa_rotation_spatial"
NUM_SAMPLES = 100
SEED = 42
THUMB_SIZE = (256, 256)

PROJECT = "thinkmorph-orbit-annotations"


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


def load_thumbnail(path):
    """Load image and resize to thumbnail for W&B upload."""
    img = Image.open(path).convert("RGB")
    img.thumbnail(THUMB_SIZE)
    return img


def main():
    print(f"Loading rotation samples from {ROTATION_QA_DIR}...")
    all_samples = load_all_rotation_samples(ROTATION_QA_DIR)
    print(f"Found {len(all_samples)} total rotation samples.")

    random.seed(SEED)
    sampled = random.sample(all_samples, min(NUM_SAMPLES, len(all_samples)))

    columns = ["id", "question_type", "question", "correct_answer", "images"]

    print("Initializing W&B run...")
    run = wandb.init(project=PROJECT, name="rotation-sample-100-preview")

    table = wandb.Table(columns=columns)
    img_keys = ["image_1", "image_2", "image_3", "image_4", "bridge_panorama"]

    for i, sample in enumerate(sampled):
        wb_images = []
        for key in img_keys:
            filename = sample["images"].get(key)
            if not filename:
                continue
            path = os.path.join(sample["novel_qa_dir"], filename)
            if os.path.exists(path):
                try:
                    wb_images.append(wandb.Image(load_thumbnail(path)))
                except Exception as e:
                    print(f"Warning: Could not load image {path}: {e}")
            else:
                print(f"Warning: Image not found: {path}")

        table.add_data(
            sample["sample_id"],
            sample["question_type"],
            sample["question"],
            sample["correct_answer"],
            wb_images,
        )
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(sampled)} entries")

    print(f"Uploading table with {len(table.data)} rows to W&B...")
    run.log({"samples": table})

    wandb.finish()
    print(f"Done! View results at: {run.url}")


if __name__ == "__main__":
    main()
