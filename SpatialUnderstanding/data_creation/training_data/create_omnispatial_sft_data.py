#!/usr/bin/env python3
"""
Convert OmniSpatial training data into ShareGPT JSONL format for SFT.

Each entry in data.json becomes one JSONL line with:
  - conversations: [{from: human, value: <system_prompt>\n\n<image>\n<question>\n<options>},
                    {from: gpt,   value: <think></think><answer>X</answer>}]
  - image: ["<task_type>/<num>.png"]  (relative to omnispatial_root)
  - id: "<original_id>"

Usage:
    python SpatialUnderstanding/data_creation/training_data/create_omnispatial_sft_data.py
    python SpatialUnderstanding/data_creation/training_data/create_omnispatial_sft_data.py \\
        --omnispatial_root /path/to/scratch/datasets/OmniSpatial/OmniSpatial-train \\
        --sample_count 2000 \\
        --output_file /path/to/scratch/VisualCoT/training_data/omnispatial_sft/omnispatial_sft.jsonl
"""

import argparse
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from tqdm import tqdm


INTERLEAVED_THINK_SYSTEM_PROMPT = (
    "Let's think step by step to answer the question. "
    "For text-based thinking, enclose the process within <think> </think>, "
    "e.g. <think> thinking process here </think>. "
    "For visual thinking, enclose the content within <image_start> </image_end>, "
    "e.g. <image_start> thinking image here </image_end>. "
    "Finally conclude with the final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)


def process_omnispatial_item(sample: dict, omnispatial_root: str):
    """Convert one OmniSpatial entry into ShareGPT format."""
    try:
        sample_id = str(sample.get("id", ""))
        question_raw = sample.get("question", "")
        if isinstance(question_raw, list):
            question_raw = " ".join(question_raw)
        question = question_raw.strip()
        task_type = sample.get("task_type", "")
        answer_idx = sample.get("answer")

        if not question or answer_idx is None or not task_type:
            print(f"[WARNING] Missing fields for sample {sample_id}")
            return None

        # Image path: <task_type>/<base_num>.png
        base_num = sample_id.split("_")[0]
        rel_image_path = os.path.join(task_type, f"{base_num}.png")
        abs_image_path = os.path.join(omnispatial_root, rel_image_path)

        if not os.path.exists(abs_image_path):
            print(f"[WARNING] Image not found: {abs_image_path}")
            return None

        # Filter empty options
        options = [o for o in sample.get("options", []) if o.strip()]
        if not options or answer_idx >= len(options):
            print(f"[WARNING] Invalid options for sample {sample_id}")
            return None

        options_str = "\n".join(f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options))
        answer_letter = chr(65 + answer_idx)

        human_value = f"{INTERLEAVED_THINK_SYSTEM_PROMPT}\n\n<image>\n{question}\n\n{options_str}"
        gpt_value = f"<think></think><answer>{answer_letter}</answer>"

        return {
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": gpt_value},
            ],
            "image": [rel_image_path],
            "id": sample_id,
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {sample.get('id', 'unknown')}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Create OmniSpatial SFT JSONL data")
    parser.add_argument(
        "--omnispatial_root",
        default="/path/to/scratch/datasets/OmniSpatial/OmniSpatial-train",
        help="Root directory of OmniSpatial-train (contains data.json and task subdirs)",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=1500,
        help="Number of samples to use (-1 = all)",
    )
    parser.add_argument(
        "--output_file",
        default="/path/to/scratch/VisualCoT/training_data/omnispatial_sft/omnispatial_sft.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    data_file = os.path.join(args.omnispatial_root, "data.json")
    print(f"Loading {data_file}")
    with open(data_file) as f:
        all_samples = json.load(f)
    print(f"Loaded {len(all_samples)} samples")

    if args.sample_count > 0 and args.sample_count < len(all_samples):
        all_samples = random.sample(all_samples, args.sample_count)
        print(f"Randomly sampled {len(all_samples)} samples")

    process_fn = partial(process_omnispatial_item, omnispatial_root=args.omnispatial_root)

    print(f"Processing with {args.max_workers} workers...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(tqdm(executor.map(process_fn, all_samples), total=len(all_samples)))

    data = [r for r in results if r is not None]
    print(f"Successfully processed {len(data)} / {len(all_samples)} samples")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Written to {args.output_file}")

    # Summary
    task_counts = Counter(
        item["image"][0].split("/")[0] for item in data
    )
    summary = {
        "total": len(data),
        "per_task_type": dict(task_counts),
        "output_file": args.output_file,
    }
    summary_file = args.output_file.replace(".jsonl", "_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary}")
    print(f"Summary written to {summary_file}")


if __name__ == "__main__":
    main()
