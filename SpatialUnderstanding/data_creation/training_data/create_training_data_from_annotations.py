#!/usr/bin/env python3
"""
Convert Gemini annotation JSON to training dataset format (parquet files).

This script reads annotations from the Gemini annotator and converts them
to the training format used by ThinkMorph.

Supports four thinking modes:
  --thinking_mode interleaved  (default)
    Text reasoning + generated image + more text reasoning + answer.
    Images: [problem_img_0, problem_img_1, reasoning_img_0]
    Output: [<think>thought_0</think><image_start>, <image_end><think>thought_1</think><answer>]

  --thinking_mode text_only
    Text reasoning only (no generated reasoning image).
    Images: [problem_img_0, problem_img_1]
    Output: [<think>full_reasoning</think><answer>]

  --thinking_mode visual_only
    Generated reasoning image only (no text reasoning).
    Images: [problem_img_0, problem_img_1, reasoning_img_0]
    Output: [<image_start>, <image_end><answer>]

  --thinking_mode no_thinking
    Direct answer without any reasoning.
    Images: [problem_img_0, problem_img_1]
    Output: [<answer>]

Image mapping:
- problem_image_0: user_1_image_local_path (first view)
- problem_image_1: user_2_image_local_path (second view)
- reasoning_image_0: center_view_output_path (center view used for reasoning)
"""

import argparse
import json
import os
import io
import random
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import pyarrow as pa
import pyarrow.parquet as pq


# --- System prompts for each thinking mode ---

INTERLEAVED_THINK_SYSTEM_PROMPT = (
    "Let's think step by step to answer the question. "
    "For text-based thinking, enclose the process within <think> </think>, "
    "e.g. <think> thinking process here </think>. "
    "For visual thinking, enclose the content within <image_start> </image_end>, "
    "e.g. <image_start> thinking image here </image_end>. "
    "Finally conclude with the final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)

TEXT_ONLY_THINK_SYSTEM_PROMPT = (
    "Let's think step by step to answer the question. "
    "Enclose your thinking process within <think> </think> tags. "
    "Finally conclude with the final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)

VISUAL_ONLY_THINK_SYSTEM_PROMPT = (
    "Think visually to answer the question. "
    "Enclose your visual thinking within <image_start> </image_end>, "
    "e.g. <image_start> thinking image here </image_end>. "
    "Finally conclude with the final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)

NO_THINKING_SYSTEM_PROMPT = (
    "Answer the question directly. "
    "Provide your final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)


def image_to_bytes(image_path: str) -> bytes:
    """Load image from path and convert to PNG bytes."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            byte_stream = io.BytesIO()
            img.save(byte_stream, format='PNG')
            return byte_stream.getvalue()
    except Exception as e:
        print(f"[ERROR] Failed to load image {image_path}: {e}")
        return None


def clean_answer_tags(text: str) -> str:
    """Remove everything starting from <answer> tag (and variants) to end of text."""
    text = re.sub(r'</?\\*answer>.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text


def parse_annotation(annotation: str, answer: str):
    """
    Parse the annotation to extract reasoning thoughts before and after <view_token>.

    The annotation format is:
    Step 1: ... reasoning before viewing center image ...
    <view_token>
    Step 2: ... reasoning after viewing center image ...
    Step 3: ... final reasoning ...
    <answer>...</answer>

    Returns:
        reasoning_thought_0: Text before <view_token>
        reasoning_thought_1: Text after <view_token> but before <answer>
        extracted_answer: The answer from <answer> tags (or use provided answer)
    """
    # Split by <view_token>
    parts = annotation.split('<view_token>')

    if len(parts) >= 2:
        reasoning_thought_0 = parts[0].strip()
        after_view = parts[1].strip()
    else:
        # No <view_token> found, put everything in thought_0
        reasoning_thought_0 = annotation.strip()
        after_view = ""

    # Extract answer from <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', after_view, re.DOTALL | re.IGNORECASE)
    if answer_match:
        extracted_answer = answer_match.group(1).strip()
    else:
        extracted_answer = answer

    # Clean all answer tags from reasoning parts
    reasoning_thought_0 = clean_answer_tags(reasoning_thought_0).strip()
    reasoning_thought_1 = clean_answer_tags(after_view).strip()

    return reasoning_thought_0, reasoning_thought_1, extracted_answer


def remap_path(path: str) -> str:
    """Remap paths from home directories to network scratch."""
    if path is None:
        return None
    path = path.replace(
        "/path/to/scratch",
        "/path/to/scratch"
    )
    path = path.replace(
        "/path/to/scratch",
        "/path/to/scratch"
    )
    return path


def _get_common_fields(result: dict):
    """Extract common fields shared across all thinking modes.

    Supports two annotation types:
    - counting/SG: has center_view_output_path, images always user_1 then user_2
    - rel/topdown: has dominant_agent, image order depends on which agent is dominant

    Returns (problem_img_0_path, problem_img_1_path, reasoning_img_path,
             question, options_text, answer_letter,
             reasoning_thought_0, reasoning_thought_1)
    or None on failure.
    """
    original_metadata = result.get('original_metadata', {})

    user_1_image_path = remap_path(original_metadata.get('user_1_image_local_path'))
    user_2_image_path = remap_path(original_metadata.get('user_2_image_local_path'))

    # Determine image ordering and reasoning image based on annotation type
    dominant_agent = result.get('dominant_agent')
    if dominant_agent is not None:
        # rel/topdown annotations: order by dominant agent
        if dominant_agent == 2:
            problem_img_0_path = user_2_image_path
            problem_img_1_path = user_1_image_path
            reasoning_img_path = remap_path(original_metadata.get('user_2_top_down_view'))
        else:
            problem_img_0_path = user_1_image_path
            problem_img_1_path = user_2_image_path
            reasoning_img_path = remap_path(original_metadata.get('user_1_top_down_view'))
    else:
        # counting/SG annotations: always user_1, user_2, center_view
        problem_img_0_path = user_1_image_path
        problem_img_1_path = user_2_image_path
        reasoning_img_path = remap_path(original_metadata.get('center_view_output_path'))

    question = result.get('question', '')
    # Strip options already embedded in the question to avoid duplication
    question = re.split(r'\n\s*Options[:\s]', question)[0].strip()
    annotation = result.get('annotation', '')

    # Get options and ground truth answer index
    if original_metadata.get("options_user_2") is not None:
        options = original_metadata["options_user_2"]
        correct_answer_idx = original_metadata["user_2_gt_answer_idx"]
    else:
        options = original_metadata.get("options_user_1", [])
        correct_answer_idx = original_metadata.get("user_1_gt_answer_idx", 0)

    if not annotation:
        print(f"[WARNING] Empty annotation for sample {result.get('sample_id')}")
        return None

    if not options or correct_answer_idx is None:
        print(f"[WARNING] Missing options or answer index for sample {result.get('sample_id')}")
        return None

    option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    options_list = [f"{option_letters[i]}. {opt}" for i, opt in enumerate(options)]
    options_text = "\nOptions: " + ", ".join(options_list)
    answer_letter = option_letters[correct_answer_idx]

    reasoning_thought_0, reasoning_thought_1, _ = parse_annotation(annotation, '')

    return (problem_img_0_path, problem_img_1_path, reasoning_img_path,
            question, options_text, answer_letter,
            reasoning_thought_0, reasoning_thought_1)


# ---------- per-mode process_item functions ----------

def process_item_interleaved(result: dict):
    """Interleaved thinking: text + image + text + answer."""
    try:
        fields = _get_common_fields(result)
        if fields is None:
            return None

        (problem_img_0_path, problem_img_1_path, reasoning_img_path,
         question, options_text, answer_letter,
         reasoning_thought_0, reasoning_thought_1) = fields

        if not all([problem_img_0_path, problem_img_1_path, reasoning_img_path]):
            print(f"[WARNING] Missing image paths for sample {result.get('sample_id')}")
            return None

        problem_img0_bytes = image_to_bytes(problem_img_0_path)
        problem_img1_bytes = image_to_bytes(problem_img_1_path)
        reasoning_img0_bytes = image_to_bytes(reasoning_img_path)

        if not all([problem_img0_bytes, problem_img1_bytes, reasoning_img0_bytes]):
            print(f"[WARNING] Failed to load images for sample {result.get('sample_id')}")
            return None

        return {
            "image_list": [problem_img0_bytes, problem_img1_bytes, reasoning_img0_bytes],
            "instruction_list": [INTERLEAVED_THINK_SYSTEM_PROMPT + "\n" + question + options_text],
            "output_text_list": [
                "<think>" + reasoning_thought_0 + "</think><image_start>",
                "<image_end><think>" + reasoning_thought_1 + "</think><answer>" + answer_letter + "</answer>"
            ],
            "sample_id": result.get('sample_id', ''),
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {result.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_item_text_only(result: dict, image_root_dir: str = "/network/scratch"):
    """Text-only thinking: ShareGPT/JSONL format for vlm_sft dataset.

    Returns dict with conversations (ShareGPT) + relative image paths.
    Images are referenced via <image> tokens and loaded by SftJSONLIterableDataset
    which does os.path.join(data_dir, image_path).
    """
    try:
        fields = _get_common_fields(result)
        if fields is None:
            return None

        (problem_img_0_path, problem_img_1_path, _reasoning_img_path,
         question, options_text, answer_letter,
         reasoning_thought_0, reasoning_thought_1) = fields

        if not all([problem_img_0_path, problem_img_1_path]):
            print(f"[WARNING] Missing image paths for sample {result.get('sample_id')}")
            return None

        # Verify images exist
        for img_path in [problem_img_0_path, problem_img_1_path]:
            if not os.path.exists(img_path):
                print(f"[WARNING] Image not found: {img_path}")
                return None

        # Convert absolute paths to relative paths w.r.t. image_root_dir
        rel_img_0 = os.path.relpath(problem_img_0_path, image_root_dir)
        rel_img_1 = os.path.relpath(problem_img_1_path, image_root_dir)

        # Combine both reasoning phases into one text block
        full_reasoning = reasoning_thought_0
        if reasoning_thought_1:
            full_reasoning += "\n\n" + reasoning_thought_1

        # Build ShareGPT format: <image> tokens reference images in order
        user_message = f"{TEXT_ONLY_THINK_SYSTEM_PROMPT}\n\n<image><image>\n{question}{options_text}"
        assistant_response = f"<think>{full_reasoning}</think><answer>{answer_letter}</answer>"

        return {
            "conversations": [
                {"from": "human", "value": user_message},
                {"from": "gpt", "value": assistant_response}
            ],
            "image": [rel_img_0, rel_img_1],
            "id": result.get('sample_id', ''),
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {result.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_item_visual_only(result: dict):
    """Visual-only thinking: reasoning image only (no text reasoning), then answer."""
    try:
        fields = _get_common_fields(result)
        if fields is None:
            return None

        (problem_img_0_path, problem_img_1_path, reasoning_img_path,
         question, options_text, answer_letter,
         _reasoning_thought_0, _reasoning_thought_1) = fields

        if not all([problem_img_0_path, problem_img_1_path, reasoning_img_path]):
            print(f"[WARNING] Missing image paths for sample {result.get('sample_id')}")
            return None

        problem_img0_bytes = image_to_bytes(problem_img_0_path)
        problem_img1_bytes = image_to_bytes(problem_img_1_path)
        reasoning_img0_bytes = image_to_bytes(reasoning_img_path)

        if not all([problem_img0_bytes, problem_img1_bytes, reasoning_img0_bytes]):
            print(f"[WARNING] Failed to load images for sample {result.get('sample_id')}")
            return None

        return {
            "image_list": [problem_img0_bytes, problem_img1_bytes, reasoning_img0_bytes],
            "instruction_list": [VISUAL_ONLY_THINK_SYSTEM_PROMPT + "\n" + question + options_text],
            "output_text_list": [
                "<image_start>",
                "<image_end><answer>" + answer_letter + "</answer>"
            ],
            "sample_id": result.get('sample_id', ''),
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {result.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_item_no_thinking(result: dict, image_root_dir: str = "/network/scratch"):
    """No thinking: direct answer without any reasoning, ShareGPT/JSONL format."""
    try:
        fields = _get_common_fields(result)
        if fields is None:
            return None

        (problem_img_0_path, problem_img_1_path, _reasoning_img_path,
         question, options_text, answer_letter,
         _reasoning_thought_0, _reasoning_thought_1) = fields

        if not all([problem_img_0_path, problem_img_1_path]):
            print(f"[WARNING] Missing image paths for sample {result.get('sample_id')}")
            return None

        # Verify images exist
        for img_path in [problem_img_0_path, problem_img_1_path]:
            if not os.path.exists(img_path):
                print(f"[WARNING] Image not found: {img_path}")
                return None

        # Convert absolute paths to relative paths w.r.t. image_root_dir
        rel_img_0 = os.path.relpath(problem_img_0_path, image_root_dir)
        rel_img_1 = os.path.relpath(problem_img_1_path, image_root_dir)

        # Build ShareGPT format: direct answer, no thinking
        user_message = f"{NO_THINKING_SYSTEM_PROMPT}\n\n<image><image>\n{question}{options_text}"
        assistant_response = f"<answer>{answer_letter}</answer>"

        return {
            "conversations": [
                {"from": "human", "value": user_message},
                {"from": "gpt", "value": assistant_response}
            ],
            "image": [rel_img_0, rel_img_1],
            "id": result.get('sample_id', ''),
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {result.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


THINKING_MODE_PROCESSORS = {
    "interleaved": process_item_interleaved,
    "text_only": process_item_text_only,
    "visual_only": process_item_visual_only,
    "no_thinking": process_item_no_thinking,
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gemini annotations to training dataset format"
    )
    parser.add_argument(
        "--annotation_file",
        nargs="+",
        default=[
            "/path/to/scratch/VisualCoT/annotations/gemini_3_pro_preview_annotations_scence_graph_spatial_v2.json",
            "/path/to/scratch/VisualCoT/annotations/gemini_3_pro_preview_annotations_scence_graph_counting.json",
            "/path/to/scratch/VisualCoT/annotations/gemini_3_pro_preview_annotations_topdown_rel.json",
        ],
        help="Path(s) to the Gemini annotation JSON file(s)"
    )
    parser.add_argument(
        "--output_dir",
        default="/path/to/scratch/VisualCoT/training_data/spatial_counting_rel",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--thinking_mode",
        choices=["interleaved", "text_only", "visual_only", "no_thinking"],
        default="no_thinking",
        help=(
            "Thinking mode for training data: "
            "interleaved = text + image + text (default), "
            "text_only = text reasoning only (no reasoning image), "
            "visual_only = reasoning image only (no text reasoning), "
            "no_thinking = direct answer without any reasoning"
        ),
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of parallel workers for processing"
    )
    parser.add_argument(
        "--rows_per_group",
        type=int,
        default=10,
        help="Rows per row group in parquet files"
    )
    parser.add_argument(
        "--groups_per_file",
        type=int,
        default=10,
        help="Row groups per parquet file"
    )
    parser.add_argument(
        "--image_root_dir",
        default="/path/to/scratch/spatial_collab_dataset/scenes",
        help=(
            "Root directory for images (text_only mode only). "
            "Image paths in JSONL will be relative to this directory. "
            "Must match data_dir in dataset_info.py."
        ),
    )

    args = parser.parse_args()

    process_fn = THINKING_MODE_PROCESSORS[args.thinking_mode]
    if args.thinking_mode in ("text_only", "no_thinking"):
        process_fn = partial(process_fn, image_root_dir=args.image_root_dir)
    print(f"Thinking mode: {args.thinking_mode}")

    # Load annotations from all files
    results = []
    for annotation_file in args.annotation_file:
        print(f"Loading annotations from {annotation_file}")
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        file_results = data.get('results', [])
        print(f"  -> {len(file_results)} results")
        results.extend(file_results)
    print(f"Loaded {len(results)} total annotation results")

    # Filter out empty annotations
    results = [r for r in results if r.get('annotation', '').strip()]
    print(f"After filtering empty annotations: {len(results)} samples")

    # Process items in parallel
    print(f"Processing samples with {args.max_workers} workers...")
    all_data = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        processed = list(tqdm(executor.map(process_fn, results), total=len(results)))
        all_data = [res for res in processed if res is not None]

    random.shuffle(all_data)
    print(f"Successfully processed {len(all_data)} samples")

    if len(all_data) == 0:
        print("No samples to write. Exiting.")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.thinking_mode in ("text_only", "no_thinking"):
        # Write JSONL for vlm_sft dataset (ShareGPT format)
        jsonl_name = "no_thinking.jsonl" if args.thinking_mode == "no_thinking" else "text_only_thinking.jsonl"
        jsonl_file = os.path.join(args.output_dir, jsonl_name)
        print(f"Writing JSONL to {jsonl_file}")
        with open(jsonl_file, 'w') as f:
            for item in all_data:
                f.write(json.dumps(item) + '\n')

        print(f"\nDone! Created {jsonl_file}")
        print(f"Total samples: {len(all_data)}")

        # Print sample for verification
        sample = all_data[0]
        print("\n--- Sample entry ---")
        print(f"ID: {sample['id']}")
        print(f"Images: {sample['image']}")
        print(f"Human: {sample['conversations'][0]['value'][:200]}...")
        print(f"GPT: {sample['conversations'][1]['value'][:200]}...")
    else:
        # Write to parquet files for interleaved / visual_only modes
        rows_per_file = args.rows_per_group * args.groups_per_file
        parquet_base_name = "chunk_"
        file_index = 0

        # Remove sample_id before writing (only for training data format)
        for item in all_data:
            if 'sample_id' in item:
                del item['sample_id']

        # Track parquet info for each file
        parquet_info = {}

        print(f"Writing parquet files to {args.output_dir}")
        for i in tqdm(range(0, len(all_data), rows_per_file)):
            file_data = all_data[i:i + rows_per_file]
            parquet_file = os.path.join(args.output_dir, f"{parquet_base_name}{file_index}.parquet")
            file_index += 1

            schema = pa.schema([
                pa.field("image_list", pa.list_(pa.binary())),
                pa.field("instruction_list", pa.list_(pa.string())),
                pa.field("output_text_list", pa.list_(pa.string())),
            ])

            num_row_groups = 0
            with pq.ParquetWriter(parquet_file, schema=schema, version="2.6") as writer:
                for j in range(0, len(file_data), args.rows_per_group):
                    group_data = file_data[j:j + args.rows_per_group]
                    group_table = pa.Table.from_pylist(group_data, schema=schema)
                    writer.write_table(group_table)
                    num_row_groups += 1

            # Record parquet info for this file
            parquet_info[parquet_file] = {
                "num_row_groups": num_row_groups,
                "num_rows": len(file_data),
            }

        # Write parquet_info.json
        parquet_info_path = os.path.join(args.output_dir, "parquet_info.json")
        with open(parquet_info_path, 'w') as f:
            json.dump(parquet_info, f, indent=2)

        print(f"\nDone! Created {file_index} parquet file(s) in {args.output_dir}")
        print(f"Parquet info saved to {parquet_info_path}")
        print(f"Total samples: {len(all_data)}")

    print(f"Thinking mode: {args.thinking_mode}")


if __name__ == "__main__":
    main()
