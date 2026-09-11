#!/usr/bin/env python3
"""
Convert Gemini annotation JSON to LLaMA-Factory/Qwen SFT format.

This script reads annotations from the Gemini annotator and converts them
to the ShareGPT format used by LLaMA-Factory for Qwen2.5-VL training.

Image mapping (input only, text-only output):
- image_0: user_1_image_local_path (first view)
- image_1: user_2_image_local_path (second view)

The annotation contains reasoning with <view_token> marking. For Qwen training,
we combine all reasoning into text-only output without visual tokens.
"""

import argparse
import json
import os
import re
from pathlib import Path
from tqdm import tqdm


SYSTEM_PROMPT_THINKING = '''Let's think step by step to answer the question. Enclose your thinking process within <think> </think> tags. Finally conclude with the final answer wrapped in <answer></answer> tags.'''

SYSTEM_PROMPT_NO_THINKING = '''Answer the question by selecting the correct option. Wrap your final answer in <answer></answer> tags.'''


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


def clean_answer_tags(text: str) -> str:
    """Remove everything starting from <answer> tag (and variants) to end of text."""
    # Remove everything from <answer> onwards (any variant)
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

    # Extract answer from <answer> tags (find the last one)
    answer_match = re.search(r'<answer>(.*?)</answer>', after_view, re.DOTALL | re.IGNORECASE)
    if answer_match:
        extracted_answer = answer_match.group(1).strip()
    else:
        extracted_answer = answer

    # Clean all answer tags from reasoning parts
    reasoning_thought_0 = clean_answer_tags(reasoning_thought_0).strip()
    reasoning_thought_1 = clean_answer_tags(after_view).strip()

    return reasoning_thought_0, reasoning_thought_1, extracted_answer


def process_item(result: dict, no_thinking: bool = False):
    """
    Process a single annotation result into LLaMA-Factory ShareGPT format.

    Returns dict with:
    - conversations: list of {from, value} dicts
    - image: list of image paths
    """
    try:
        original_metadata = result.get('original_metadata', {})

        # Get image paths (only the two input views, no center view needed for text-only output)
        user_1_image_path = remap_path(original_metadata.get('user_1_image_local_path'))
        user_2_image_path = remap_path(original_metadata.get('user_2_image_local_path'))

        if not all([user_1_image_path, user_2_image_path]):
            print(f"[WARNING] Missing image paths for sample {result.get('sample_id')}")
            return None

        # Verify images exist
        for img_path in [user_1_image_path, user_2_image_path]:
            if not os.path.exists(img_path):
                print(f"[WARNING] Image not found: {img_path}")
                return None

        # Get question, answer, and options from metadata
        question = result.get('question', '')
        annotation = result.get('annotation', '')

        # Get options and ground truth answer index
        # Prefer user_2 options if available, otherwise fall back to user_1
        if result['original_metadata'].get("options_user_2") is not None:
            options = result['original_metadata']["options_user_2"]
            correct_answer_idx = result['original_metadata']["user_2_gt_answer_idx"]
        else:
            options = result['original_metadata'].get("options_user_1", [])
            correct_answer_idx = result['original_metadata'].get("user_1_gt_answer_idx", 0)

        if not annotation:
            print(f"[WARNING] Empty annotation for sample {result.get('sample_id')}")
            return None

        # Parse the annotation to get reasoning (answer will come from ground truth index)
        reasoning_thought_0, reasoning_thought_1, _ = parse_annotation(annotation, result.get('answer', ''))

        # Format the options as a list (A, B, C, D)
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        if not options or correct_answer_idx is None:
            print(f"[WARNING] Missing options or answer index for sample {result.get('sample_id')}")
            return None

        options_list = [f"{option_letters[i]}. {opt}" for i, opt in enumerate(options)]
        options_text = "\nOptions: " + ", ".join(options_list)

        # Get answer letter from ground truth index
        answer_letter = option_letters[correct_answer_idx]

        # Build the user message with system prompt, images, question, and options
        # Images are referenced with <image> tokens in order
        if no_thinking:
            system_prompt = SYSTEM_PROMPT_NO_THINKING
            assistant_response = f"<answer>{answer_letter}</answer>"
        else:
            system_prompt = SYSTEM_PROMPT_THINKING
            # Build the assistant response with text-only reasoning (no visual tokens)
            # Combine both reasoning parts into a single think block
            full_reasoning = reasoning_thought_0
            if reasoning_thought_1:
                full_reasoning = f"{reasoning_thought_0}\n\n{reasoning_thought_1}"
            assistant_response = f"<think>{full_reasoning}</think><answer>{answer_letter}</answer>"

        user_message = f"{system_prompt}\n\n<image><image>\n{question}{options_text}"

        return {
            "conversations": [
                {"from": "human", "value": user_message},
                {"from": "gpt", "value": assistant_response}
            ],
            # Only include the two input images (no center view since we're not generating it)
            "image": [user_1_image_path, user_2_image_path],
            "id": result.get('sample_id', '')
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {result.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gemini annotations to LLaMA-Factory/Qwen SFT format"
    )
    parser.add_argument(
        "--annotation_files",
        nargs="+",
        default=[
            "/path/to/scratch/VisualCoT/annotations/gemini_3_pro_preview_annotations_scence_graph.json",
            "/path/to/scratch/VisualCoT/annotations/gemini_3_pro_preview_annotations_scence_graph_counting.json",
        ],
        help="Path(s) to one or more Gemini annotation JSON files"
    )
    parser.add_argument(
        "--output_file",
        default="/path/to/scratch/VisualCoT/training_data/spatial_counting_SG_qwen_sft.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--no_thinking",
        action="store_true",
        help="Skip thinking/reasoning; output only <answer> tags"
    )
    parser.add_argument(
        "--check_images",
        action="store_true",
        help="Check if all images exist (slower but validates data)"
    )

    args = parser.parse_args()

    # Load annotations from all files
    results = []
    for annotation_file in args.annotation_files:
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

    # Process items
    print("Processing samples...")
    all_data = []
    skipped = 0
    for result in tqdm(results):
        processed = process_item(result, no_thinking=args.no_thinking)
        if processed is not None:
            all_data.append(processed)
        else:
            skipped += 1

    print(f"Successfully processed {len(all_data)} samples")
    print(f"Skipped {skipped} samples")

    if len(all_data) == 0:
        print("No samples to write. Exiting.")
        return

    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write to JSON file
    with open(args.output_file, 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"\nDone! Created {args.output_file}")
    print(f"Total samples: {len(all_data)}")

    # Print sample for verification
    print("\n--- Sample entry ---")
    sample = all_data[0]
    print(f"ID: {sample['id']}")
    print(f"Images: {sample['image']}")
    print(f"Human: {sample['conversations'][0]['value'][:200]}...")
    print(f"GPT: {sample['conversations'][1]['value'][:200]}...")


if __name__ == "__main__":
    main()
