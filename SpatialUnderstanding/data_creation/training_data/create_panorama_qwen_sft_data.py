#!/usr/bin/env python3
"""
Convert panorama_train_samples_filtered.json to LLaMA-Factory/Qwen SFT format.

Input fields used per sample:
  - user_1_image_local_path, user_2_image_local_path
  - question_both_views
  - options_user_1 / options_user_2 (prefer user_2 if not None)
  - user_1_gt_answer_idx / user_2_gt_answer_idx
  - correct_answer (fallback if no options)

Output: ShareGPT JSON array suitable for LLaMA-Factory training.

Usage:
    python SpatialUnderstanding/data_creation/training_data/create_panorama_qwen_sft_data.py
    python SpatialUnderstanding/data_creation/training_data/create_panorama_qwen_sft_data.py \\
        --train_file /path/to/scratch/VisualCoT/training_data/panorama_train_samples_filtered.json \\
        --output_file /path/to/scratch/VisualCoT/training_data/panorama_qwen_sft.json
"""

import argparse
import json
import os

from tqdm import tqdm


SYSTEM_PROMPT_NO_THINKING = (
    "Answer the question by selecting the correct option. "
    "Wrap your final answer in <answer></answer> tags."
)


def remap_path(path: str) -> str:
    """Remap paths from home directories to network scratch."""
    if path is None:
        return None
    path = path.replace(
        "/path/to/scratch",
        "/path/to/scratch",
    )
    path = path.replace(
        "/path/to/scratch",
        "/path/to/scratch",
    )
    return path


def process_item(sample: dict):
    """Convert one panorama sample into LLaMA-Factory ShareGPT format."""
    try:
        sample_id = sample.get("sample_id", "")

        img1 = remap_path(sample.get("user_1_image_local_path"))
        img2 = remap_path(sample.get("user_2_image_local_path"))

        if not img1 or not img2:
            print(f"[WARNING] Missing image paths for {sample_id}")
            return None
        for p in [img1, img2]:
            if not os.path.exists(p):
                print(f"[WARNING] Image not found: {p}")
                return None

        question = (sample.get("question_both_views") or "").strip()
        if not question:
            print(f"[WARNING] Empty question for {sample_id}")
            return None

        # Prefer user_2 options/answer if available
        if sample.get("options_user_2") is not None:
            options = sample["options_user_2"]
            answer_idx = sample.get("user_2_gt_answer_idx")
        else:
            options = sample.get("options_user_1") or []
            answer_idx = sample.get("user_1_gt_answer_idx")

        option_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]

        if options and answer_idx is not None:
            options_list = [f"{option_letters[i]}. {opt}" for i, opt in enumerate(options)]
            options_text = "\nOptions: " + ", ".join(options_list)
            answer_value = option_letters[answer_idx]
        else:
            correct_answer = (sample.get("correct_answer") or "").strip()
            if not correct_answer:
                print(f"[WARNING] No options or correct_answer for {sample_id}")
                return None
            options_text = ""
            answer_value = correct_answer

        user_message = f"{SYSTEM_PROMPT_NO_THINKING}\n\n<image><image>\n{question}{options_text}"
        assistant_response = f"<answer>{answer_value}</answer>"

        return {
            "conversations": [
                {"from": "human", "value": user_message},
                {"from": "gpt", "value": assistant_response},
            ],
            "image": [img1, img2],
            "id": sample_id,
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {sample.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert panorama samples to LLaMA-Factory/Qwen SFT format"
    )
    parser.add_argument(
        "--train_file",
        default="/path/to/scratch/VisualCoT/training_data/panorama_train_samples_filtered.json",
        help="Path to panorama_train_samples_filtered.json",
    )
    parser.add_argument(
        "--output_file",
        default="/path/to/scratch/VisualCoT/training_data/panorama_qwen_sft.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    print(f"Loading {args.train_file}")
    with open(args.train_file) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples")

    all_data = []
    skipped = 0
    for sample in tqdm(samples):
        result = process_item(sample)
        if result is not None:
            all_data.append(result)
        else:
            skipped += 1

    print(f"Successfully processed {len(all_data)} samples, skipped {skipped}")

    if not all_data:
        print("No samples to write. Exiting.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Written to {args.output_file}")

    s = all_data[0]
    print("\n--- Sample entry ---")
    print(f"ID: {s['id']}")
    print(f"Images: {s['image']}")
    print(f"Human: {s['conversations'][0]['value'][:200]}...")
    print(f"GPT: {s['conversations'][1]['value']}")


if __name__ == "__main__":
    main()
