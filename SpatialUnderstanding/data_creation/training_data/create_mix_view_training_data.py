#!/usr/bin/env python3
"""
Create training data from topdown map QA samples.

Input: Two perspective images + question, with rendered topdown map as
the visual reasoning image.

Supports five thinking modes:
  --thinking_mode BEV_visual_only (default)
    BEV relative distance image as visual reasoning. Fixed <BEV> mode token.
    Images: [user_1_image, user_2_image, bev_relative_distance]
    Output: ["<BEV> <image_start>", "<image_end><answer>LETTER</answer>"]
    Format: Parquet files

  --thinking_mode full_BEV
    Full topdown map as visual reasoning image. Fixed <BEV> mode token.
    Images: [user_1_image, user_2_image, topdown_agent_2]
    Output: ["<BEV> <image_start>", "<image_end><answer>LETTER</answer>"]
    Format: Parquet files

  --thinking_mode panorama_only
    Always uses rendered panorama image as visual reasoning. Fixed <panoramic> mode token.
    Images: [user_1_image, user_2_image, panorama_image]
    Output: ["<panoramic> <image_start>", "<image_end><answer>LETTER</answer>"]
    Format: Parquet files

  --thinking_mode dynamic_visual_only
    Mixed modes: selects panorama or BEV image based on question_type.
    anchor/counting/spatial_orientation → <panoramic> (1024, rendered panorama)
    closest/farthest                    → <BEV>       (720x720, BEV relative distance)
    Images: [user_1_image, user_2_image, thinking_image]
    Output: ["<MODE> <image_start>", "<image_end><answer>LETTER</answer>"]
    Format: Parquet files

  --thinking_mode no_thinking
    Direct answer without any reasoning.
    Images: [user_1_image, user_2_image]
    Output: ShareGPT JSONL with <answer>LETTER</answer>

Usage:
    python SpatialUnderstanding/data_creation/training_data/create_mix_view_training_data.py --thinking_mode BEV_visual_only
    python SpatialUnderstanding/data_creation/training_data/create_mix_view_training_data.py --thinking_mode full_BEV
    python SpatialUnderstanding/data_creation/training_data/create_mix_view_training_data.py --thinking_mode panorama_only
    python SpatialUnderstanding/data_creation/training_data/create_mix_view_training_data.py --thinking_mode dynamic_visual_only
    python SpatialUnderstanding/data_creation/training_data/create_mix_view_training_data.py --thinking_mode no_thinking
"""

import argparse
import io
import json
import os
import random
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


BEV_VISUAL_ONLY_THINK_SYSTEM_PROMPT = (
    "Think visually to answer the question. "
    "For visual thinking, output the mode token before the image block: "
    "<BEV> <image_start> thinking image here <image_end>."
)

PANORAMA_ONLY_THINK_SYSTEM_PROMPT = (
    "Think visually to answer the question. "
    "For visual thinking using panoramic view (wide-angle): "
    "<panoramic> <image_start> thinking image here <image_end>."
)

DYNAMIC_VISUAL_ONLY_THINK_SYSTEM_PROMPT = (
    "Think visually to answer the question. "
    "Output the mode token before the image block. "
    "For panoramic reasoning (wide-angle view): <panoramic> <image_start> ... <image_end>. "
    "For bird's-eye-view spatial reasoning (top-down): <BEV> <image_start> ... <image_end>."
    "Finally conclude with the final answer wrapped in <answer></answer> tags, "
    "i.e. <answer> answer here </answer>."
)

NO_THINKING_SYSTEM_PROMPT = (
    "Answer the question directly. "
    "Provide your final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)

# ROTATION_BEV_THINK_SYSTEM_PROMPT = (
#     "Think visually to answer the question. "
#     "You are given four images (image 1, 2, 3, image 4) showing the same location from "
#     "four wall-facing directions (front, right, back, left). "
#     "For visual thinking using bird's-eye-view spatial reasoning (top-down): "
#     "<BEV> <image_start> thinking image here <image_end>."
# )

# Mapping from question_type to the appropriate dynamic visual mode token.
#   anchor / counting / spatial_orientation → <panoramic> (720×1024, wide-angle view)
#   closest / farthest                      → <BEV>       (720×720, top-down distance map)
QUESTION_TYPE_TO_MODE = {
    'anchor':               '<panoramic>',
    'counting':             '<panoramic>',
    'spatial_orientation':  '<panoramic>',
    'closest':              '<BEV>',
    'farthest':             '<BEV>',
}


def image_to_bytes(image_path: str) -> bytes:
    """Load image from path and convert to PNG bytes."""
    try:
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            byte_stream = io.BytesIO()
            img.save(byte_stream, format='PNG')
            return byte_stream.getvalue()
    except Exception as e:
        print(f"[ERROR] Failed to load image {image_path}: {e}")
        return None


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


def resolve_topdown_path(sample: dict, topdown_dir: str) -> str:
    """Resolve the rendered topdown map path for a sample (agent_2 view)."""
    scene_id = sample.get('scene_id')
    sample_id = sample.get('sample_id')
    if not scene_id or not sample_id:
        return None
    return os.path.join(topdown_dir, scene_id, f"topdown_agent_2_{sample_id}.png")


def resolve_panorama_path(sample: dict, panorama_dir: str) -> str:
    """Resolve the rendered panorama path for a sample.

    The render script deduplicates by scene_id, so a scene shared by multiple
    question types may only have one panorama file (named after the first
    sample_id rendered). Fall back to any panorama_blender_limits_*.png in
    the scene directory if the exact sample_id file is missing.
    """
    scene_id = sample.get('scene_id')
    sample_id = sample.get('sample_id')
    if not scene_id or not sample_id:
        return None
    exact = os.path.join(panorama_dir, scene_id, f"panorama_blender_limits_{sample_id}.png")
    if os.path.exists(exact):
        return exact
    # Fallback: find any panorama file for this scene
    scene_dir = os.path.join(panorama_dir, scene_id)
    if os.path.isdir(scene_dir):
        for fname in os.listdir(scene_dir):
            if fname.startswith("panorama_blender_limits_") and fname.endswith(".png"):
                return os.path.join(scene_dir, fname)
    return exact  # return non-existent path so caller can report it missing


def resolve_bev_path(sample: dict, bev_dir: str) -> str:
    """Resolve the BEV relative distance image path for a sample."""
    scene_id = sample.get('scene_id')
    sample_id = sample.get('sample_id')
    if not scene_id or not sample_id:
        return None
    # sample_id like "relative_distance_014670" -> numeric suffix "014670"
    numeric_part = sample_id.split('_')[-1]
    return os.path.join(bev_dir, scene_id, f"bev_relative_distance_{numeric_part}.png")


def resolve_dynamic_thinking_image_path(sample: dict, panorama_dir: str, bev_dir: str) -> str:
    """Dispatch to BEV for closest/farthest, panorama for all other question types."""
    question_type = sample.get('question_type', '')
    if question_type in ('closest', 'farthest'):
        return resolve_bev_path(sample, bev_dir)
    else:
        return resolve_panorama_path(sample, panorama_dir)


def _get_answer_and_options(sample: dict):
    """Extract answer text and options list from a sample."""
    options_user_1 = sample.get('options_user_1')
    options_user_2 = sample.get('options_user_2')
    if options_user_1 is not None:
        options = options_user_1
        correct_answer_idx = sample.get('user_1_gt_answer_idx')
    elif options_user_2 is not None:
        options = options_user_2
        correct_answer_idx = sample.get('user_2_gt_answer_idx')
    else:
        options = sample.get('options', [])
        correct_answer_idx = sample.get('correct_answer_idx')

    if correct_answer_idx is None:
        answer_text = sample.get('correct_answer', '').strip()
    else:
        answer_text = chr(65 + correct_answer_idx)

    return options, answer_text


def _format_question(question: str, options: list) -> str:
    """Append lettered options to question text if options are present."""
    if not options:
        return question
    options_str = "\n".join(f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options))
    return f"{question}\n\n{options_str}"


def load_test_scene_ids(test_files: list) -> set:
    """Load all scene_ids from test JSON files (each is a list of dicts with 'scene_id')."""
    scene_ids = set()
    for fpath in test_files:
        if not os.path.exists(fpath):
            print(f"[WARNING] Test file not found, skipping: {fpath}")
            continue
        with open(fpath, 'r') as f:
            entries = json.load(f)
        for entry in entries:
            sid = entry.get('scene_id')
            if sid:
                scene_ids.add(sid)
    print(f"Loaded {len(scene_ids)} test scene IDs from {len(test_files)} files")
    return scene_ids


def load_rotation_samples(rotation_dir: str) -> list:
    """Scan rotation_dir recursively for rotation_qa_questions.json files.
    Returns sorted list of (novel_qa_dir, question_dict) pairs."""
    pairs = []
    for dirpath, dirnames, filenames in os.walk(rotation_dir):
        dirnames.sort()
        if 'rotation_qa_questions.json' in filenames:
            json_path = os.path.join(dirpath, 'rotation_qa_questions.json')
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                for q in data.get('rotation_questions', []):
                    pairs.append((dirpath, q))
            except Exception as e:
                print(f"[WARNING] Failed to load {json_path}: {e}")
    pairs.sort(key=lambda x: (x[0], x[1].get('images', {}).get('bridge_bev', '')))
    return pairs


def process_item_rotation(item: tuple) -> dict:
    """Process one rotation QA question. Takes (novel_qa_dir, question_dict)."""
    try:
        novel_qa_dir, question = item
        question_type = question.get('question_type', '')
        question_text = question.get('question', '').strip()
        options = question.get('options', [])
        correct_index = question.get('correct_index')

        if not question_text or correct_index is None:
            print(f"[WARNING] Missing question or answer in {novel_qa_dir}")
            return None

        answer_text = chr(65 + correct_index)

        images_dict = question.get('images', {})
        img1_path = os.path.join(novel_qa_dir, images_dict.get('image_1', ''))
        img2_path = os.path.join(novel_qa_dir, images_dict.get('image_2', ''))
        img3_path = os.path.join(novel_qa_dir, images_dict.get('image_3', ''))
        img4_path = os.path.join(novel_qa_dir, images_dict.get('image_4', ''))
        bev_path = os.path.join(novel_qa_dir, images_dict.get('bridge_bev', ''))

        for p in [img1_path, img2_path, img3_path, img4_path, bev_path]:
            if not p or not os.path.exists(p):
                print(f"[WARNING] Image not found: {p}")
                return None

        img1_bytes = image_to_bytes(img1_path)
        img2_bytes = image_to_bytes(img2_path)
        img3_bytes = image_to_bytes(img3_path)
        img4_bytes = image_to_bytes(img4_path)
        bev_bytes = image_to_bytes(bev_path)

        if not all([img1_bytes, img2_bytes, img3_bytes, img4_bytes, bev_bytes]):
            print(f"[WARNING] Failed to load images for rotation sample in {novel_qa_dir}")
            return None

        # MCQ: options already embedded in question text as "A. ... B. ..."
        # Yes/No: options not embedded, append them
        if question_type == 'rotation_direction_mcq':
            full_question = question_text
        else:
            full_question = _format_question(question_text, options)

        # Build sample_id from path: {ROOM_TYPE}/{SCENE_ID}/novel_qa/
        path_parts = novel_qa_dir.rstrip('/').split('/')
        # novel_qa_dir ends with .../novel_qa, so scene_id is second-to-last, room_type is third-to-last
        scene_id = path_parts[-2] if len(path_parts) >= 2 else 'unknown'
        room_type = path_parts[-3] if len(path_parts) >= 3 else 'unknown'
        bev_stem = os.path.splitext(images_dict.get('bridge_bev', 'bev'))[0]
        sample_id = f"{room_type}__{scene_id}__{question_type}__{bev_stem}"

        return {
            "image_list": [img1_bytes, img2_bytes, img3_bytes, img4_bytes, bev_bytes],
            "instruction_list": [DYNAMIC_VISUAL_ONLY_THINK_SYSTEM_PROMPT + "\n" + full_question],
            "output_text_list": [
                "<BEV> <image_start>",
                f"<image_end><answer>{answer_text}</answer>",
            ],
            "sample_id": sample_id,
        }

    except Exception as e:
        print(f"[ERROR] Failed to process rotation sample in {item[0]}: {e}")
        import traceback
        traceback.print_exc()
        return None


def apply_sampling(original_data: list, rotation_data: list, total_samples, rotation_number: int) -> list:
    """Combine original and rotation data with optional sampling controls."""
    n_rotation = min(rotation_number, len(rotation_data))
    if n_rotation < rotation_number:
        print(f"[WARNING] Requested {rotation_number} rotation samples but only {len(rotation_data)} available. Using all.")

    if total_samples is None:
        # No cap on originals: use all originals + up to rotation_number rotation samples
        sampled_rotation = random.sample(rotation_data, n_rotation) if n_rotation > 0 else []
        result = list(original_data) + sampled_rotation
        random.shuffle(result)
        print(f"Combined {len(original_data)} original + {n_rotation} rotation = {len(result)} total")
        return result

    # total_samples is set: originals fill the remainder after rotation
    n_original = min(total_samples - n_rotation, len(original_data))
    if n_original < total_samples - n_rotation:
        print(f"[WARNING] Requested {total_samples - n_rotation} original samples but only {len(original_data)} available. Using all.")

    sampled_original = random.sample(original_data, n_original) if n_original > 0 else []
    sampled_rotation = random.sample(rotation_data, n_rotation) if n_rotation > 0 else []

    result = sampled_original + sampled_rotation
    random.shuffle(result)
    print(f"Sampled {n_original} original + {n_rotation} rotation = {len(result)} total")
    return result


def process_item_bev_only(sample: dict, bev_dir: str):
    """BEV-only visual thinking: uses BEV relative distance image with fixed <BEV> mode token."""
    try:
        sample_id = sample.get('sample_id', '')
        question = sample.get('question_both_views', '').strip()
        options, answer_text = _get_answer_and_options(sample)

        if not question or not answer_text:
            print(f"[WARNING] Missing question or answer for sample {sample_id}")
            return None

        img1_path = remap_path(sample.get('user_1_image_local_path'))
        img2_path = remap_path(sample.get('user_2_image_local_path'))
        bev_path = resolve_bev_path(sample, bev_dir)

        if not all([img1_path, img2_path, bev_path]):
            return None
        if not os.path.exists(bev_path):
            print(f"[WARNING] BEV image not found: {bev_path}")
            return None

        img1_bytes = image_to_bytes(img1_path)
        img2_bytes = image_to_bytes(img2_path)
        bev_bytes = image_to_bytes(bev_path)

        if not all([img1_bytes, img2_bytes, bev_bytes]):
            print(f"[WARNING] Failed to load images for sample {sample_id}")
            return None

        return {
            "image_list": [img1_bytes, img2_bytes, bev_bytes],
            "instruction_list": [BEV_VISUAL_ONLY_THINK_SYSTEM_PROMPT + "\n" + _format_question(question, options)],
            "output_text_list": [
                "<BEV> <image_start>",
                f"<image_end><answer>{answer_text}</answer>",
            ],
            "sample_id": sample_id,
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {sample.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_item_full_bev(sample: dict, topdown_dir: str):
    """Full BEV visual thinking: uses rendered topdown map with fixed <BEV> mode token."""
    try:
        sample_id = sample.get('sample_id', '')
        question = sample.get('question_both_views', '').strip()
        options, answer_text = _get_answer_and_options(sample)

        if not question or not answer_text:
            print(f"[WARNING] Missing question or answer for sample {sample_id}")
            return None

        img1_path = remap_path(sample.get('user_1_image_local_path'))
        img2_path = remap_path(sample.get('user_2_image_local_path'))
        topdown_path = resolve_topdown_path(sample, topdown_dir)

        if not all([img1_path, img2_path, topdown_path]):
            return None
        if not os.path.exists(topdown_path):
            print(f"[WARNING] Topdown map not found: {topdown_path}")
            return None

        img1_bytes = image_to_bytes(img1_path)
        img2_bytes = image_to_bytes(img2_path)
        topdown_bytes = image_to_bytes(topdown_path)

        if not all([img1_bytes, img2_bytes, topdown_bytes]):
            print(f"[WARNING] Failed to load images for sample {sample_id}")
            return None

        return {
            "image_list": [img1_bytes, img2_bytes, topdown_bytes],
            "instruction_list": [BEV_VISUAL_ONLY_THINK_SYSTEM_PROMPT + "\n" + _format_question(question, options)],
            "output_text_list": [
                "<BEV> <image_start>",
                f"<image_end><answer>{answer_text}</answer>",
            ],
            "sample_id": sample_id,
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {sample.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_item_panorama_only(sample: dict, panorama_dir: str):
    """Panorama-only visual thinking: uses panorama image with fixed <panoramic> mode token."""
    try:
        sample_id = sample.get('sample_id', '')
        question = sample.get('question_both_views', '').strip()
        options, answer_text = _get_answer_and_options(sample)

        if not question or not answer_text:
            print(f"[WARNING] Missing question or answer for sample {sample_id}")
            return None

        img1_path = remap_path(sample.get('user_1_image_local_path'))
        img2_path = remap_path(sample.get('user_2_image_local_path'))
        panorama_path = resolve_panorama_path(sample, panorama_dir)

        if not all([img1_path, img2_path, panorama_path]):
            return None
        if not os.path.exists(panorama_path):
            print(f"[WARNING] Panorama image not found: {panorama_path}")
            return None

        img1_bytes = image_to_bytes(img1_path)
        img2_bytes = image_to_bytes(img2_path)
        panorama_bytes = image_to_bytes(panorama_path)

        if not all([img1_bytes, img2_bytes, panorama_bytes]):
            print(f"[WARNING] Failed to load images for sample {sample_id}")
            return None

        return {
            "image_list": [img1_bytes, img2_bytes, panorama_bytes],
            "instruction_list": [PANORAMA_ONLY_THINK_SYSTEM_PROMPT + "\n" + _format_question(question, options)],
            "output_text_list": [
                "<panoramic> <image_start>",
                f"<image_end><answer>{answer_text}</answer>",
            ],
            "sample_id": sample_id,
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {sample.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_item_dynamic(sample: dict, panorama_dir: str, bev_dir: str):
    """Dynamic visual thinking: mode token and image selected by question_type."""
    try:
        sample_id = sample.get('sample_id', '')
        question = sample.get('question_both_views', '').strip()
        options, answer_text = _get_answer_and_options(sample)

        if not question or not answer_text:
            print(f"[WARNING] Missing question or answer for sample {sample_id}")
            return None

        img1_path = remap_path(sample.get('user_1_image_local_path'))
        img2_path = remap_path(sample.get('user_2_image_local_path'))
        thinking_path = resolve_dynamic_thinking_image_path(sample, panorama_dir, bev_dir)

        if not all([img1_path, img2_path, thinking_path]):
            return None
        if not os.path.exists(thinking_path):
            print(f"[WARNING] Thinking image not found: {thinking_path}")
            return None

        img1_bytes = image_to_bytes(img1_path)
        img2_bytes = image_to_bytes(img2_path)
        thinking_bytes = image_to_bytes(thinking_path)

        if not all([img1_bytes, img2_bytes, thinking_bytes]):
            print(f"[WARNING] Failed to load images for sample {sample_id}")
            return None

        question_type = sample.get('question_type', '')
        mode_token = QUESTION_TYPE_TO_MODE.get(question_type, '<panoramic>')

        return {
            "image_list": [img1_bytes, img2_bytes, thinking_bytes],
            "instruction_list": [DYNAMIC_VISUAL_ONLY_THINK_SYSTEM_PROMPT + "\n" + _format_question(question, options)],
            "output_text_list": [
                f"{mode_token} <image_start>",
                f"<image_end><answer>{answer_text}</answer>",
            ],
            "sample_id": sample_id,
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {sample.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_item_no_thinking(sample: dict, image_root_dir: str):
    """No thinking: direct answer without any reasoning, ShareGPT/JSONL format."""
    try:
        sample_id = sample.get('sample_id', '')
        question = sample.get('question_both_views', '').strip()
        options, answer_text = _get_answer_and_options(sample)

        if not question or not answer_text:
            print(f"[WARNING] Missing question or answer for sample {sample_id}")
            return None

        img1_path = remap_path(sample.get('user_1_image_local_path'))
        img2_path = remap_path(sample.get('user_2_image_local_path'))

        if not all([img1_path, img2_path]):
            return None
        for img_path in [img1_path, img2_path]:
            if not os.path.exists(img_path):
                print(f"[WARNING] Image not found: {img_path}")
                return None

        rel_img1 = os.path.relpath(img1_path, image_root_dir)
        rel_img2 = os.path.relpath(img2_path, image_root_dir)

        user_message = f"{NO_THINKING_SYSTEM_PROMPT}\n\n<image><image>\n{_format_question(question, options)}"
        return {
            "conversations": [
                {"from": "human", "value": user_message},
                {"from": "gpt", "value": f"<answer>{answer_text}</answer>"},
            ],
            "image": [rel_img1, rel_img2],
            "id": sample_id,
        }

    except Exception as e:
        print(f"[ERROR] Failed to process sample {sample.get('sample_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Create training data from topdown map QA samples"
    )
    parser.add_argument(
        "--train_file",
        default="/path/to/scratch/VisualCoT/training_data/panorama_train_samples.json",
        help="Path to panorama_train_samples.json",
    )
    parser.add_argument(
        "--topdown_dir",
        default="/path/to/scratch/VisualCoT/infinigen/topdown_maps_train",
        help="Directory containing rendered topdown maps (<scene_id>/topdown_agent_2_<sample_id>.png)",
    )
    parser.add_argument(
        "--output_dir",
        default="/path/to/scratch/VisualCoT/training_data/mix_view_qa_Cosmix_only",
        help="Output directory for parquet/JSONL files",
    )
    parser.add_argument(
        "--thinking_mode",
        choices=["BEV_visual_only", "full_BEV", "panorama_only", "dynamic_visual_only", "no_thinking"],
        default="dynamic_visual_only",
        help=(
            "Thinking mode for training data: "
            "BEV_visual_only = BEV relative distance image (fixed <BEV> token), "
            "full_BEV = rendered topdown map (fixed <BEV> token), "
            "panorama_only = panorama image only (fixed <panoramic> token), "
            "dynamic_visual_only = mixed modes by question_type "
            "(<panoramic> for anchor/counting/spatial_orientation, <BEV> for closest/farthest), "
            "no_thinking = direct answer without any reasoning"
        ),
    )
    parser.add_argument(
        "--panorama_dir",
        default="/path/to/scratch/VisualCoT/infinigen/rendered_panorama_train",
        help="Directory containing rendered panorama images (<scene_id>/panorama_blender_limits_<sample_id>.png)",
    )
    parser.add_argument(
        "--bev_dir",
        default="/path/to/scratch/VisualCoT/bev_relative_distance",
        help="Directory containing BEV relative distance images (<scene_id>/bev_relative_distance_<NNNNNN>.png)",
    )
    parser.add_argument(
        "--rotation_dir",
        default="/path/to/scratch/VisualCoT/novel_qa_rotation",
        help="Root directory for rotation QA data ({ROOM_TYPE}/{SCENE_ID}/novel_qa/rotation_qa_questions.json)",
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=None,
        help="Cap total output samples. If None, use all available samples.",
    )
    parser.add_argument(
        "--rotation_number",
        type=int,
        default=0,
        help=(
            "Number of rotation QA samples to include (0 = no rotation). "
            "If --total_samples is set, originals fill the remainder (total_samples - rotation_number). "
            "If --total_samples is not set, all originals are kept and rotation_number rotation samples are appended."
        ),
    )
    parser.add_argument(
        "--test_files",
        nargs="*",
        default=[
            "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_dataset_map_questions_normalized.json",
            "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
            "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
            "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
            "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
        ],
        help="Test set JSON files. Scenes appearing in these files are excluded from training data.",
    )
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--rows_per_group", type=int, default=10)
    parser.add_argument("--groups_per_file", type=int, default=10)
    parser.add_argument(
        "--image_root_dir",
        default="/path/to/scratch/spatial_collab_dataset/scenes",
        help="Root directory for images (no_thinking mode only).",
    )

    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.thinking_mode)
    print(f"Thinking mode: {args.thinking_mode}")
    print(f"Output dir: {output_dir}")

    # Load test scene IDs for filtering
    test_scene_ids = load_test_scene_ids(args.test_files) if args.test_files else set()

    print(f"Loading train samples from {args.train_file}")
    with open(args.train_file, 'r') as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples")

    samples = [s for s in samples if s.get('question_both_views', '').strip()]
    print(f"After filtering empty questions: {len(samples)} valid samples")

    # Filter out test set scenes
    if test_scene_ids:
        before = len(samples)
        samples = [s for s in samples if s.get('scene_id') not in test_scene_ids]
        print(f"After filtering test scenes: {len(samples)} samples ({before - len(samples)} removed)")

    # Pre-filter samples that have the required image on disk
    if args.thinking_mode == "BEV_visual_only":
        before = len(samples)
        samples = [s for s in samples if os.path.exists(resolve_bev_path(s, args.bev_dir))]
        print(f"After filtering missing BEV images: {len(samples)} samples ({before - len(samples)} filtered)")
    elif args.thinking_mode == "full_BEV":
        before = len(samples)
        samples = [s for s in samples if os.path.exists(resolve_topdown_path(s, args.topdown_dir))]
        print(f"After filtering missing topdown maps: {len(samples)} samples ({before - len(samples)} filtered)")
    elif args.thinking_mode == "panorama_only":
        before = len(samples)
        samples = [s for s in samples if os.path.exists(resolve_panorama_path(s, args.panorama_dir))]
        print(f"After filtering missing panorama images: {len(samples)} samples ({before - len(samples)} filtered)")
    elif args.thinking_mode == "dynamic_visual_only":
        from collections import Counter
        before = len(samples)
        kept, missing_panorama, missing_bev = [], [], []
        for s in samples:
            qt = s.get('question_type', '')
            path = resolve_dynamic_thinking_image_path(s, args.panorama_dir, args.bev_dir)
            if os.path.exists(path):
                kept.append(s)
            elif qt in ('closest', 'farthest'):
                missing_bev.append(qt)
            else:
                missing_panorama.append(qt)
        samples = kept
        print(f"After filtering missing thinking images: {len(samples)} samples ({before - len(samples)} filtered)")
        if missing_panorama:
            print(f"  Missing panorama images by question_type: {dict(Counter(missing_panorama))}")
        if missing_bev:
            print(f"  Missing BEV images by question_type: {dict(Counter(missing_bev))}")

    # Select processor
    if args.thinking_mode == "BEV_visual_only":
        process_fn = partial(process_item_bev_only, bev_dir=args.bev_dir)
    elif args.thinking_mode == "full_BEV":
        process_fn = partial(process_item_full_bev, topdown_dir=args.topdown_dir)
    elif args.thinking_mode == "panorama_only":
        process_fn = partial(process_item_panorama_only, panorama_dir=args.panorama_dir)
    elif args.thinking_mode == "dynamic_visual_only":
        process_fn = partial(process_item_dynamic, panorama_dir=args.panorama_dir, bev_dir=args.bev_dir)
    else:  # no_thinking
        process_fn = partial(process_item_no_thinking, image_root_dir=args.image_root_dir)

    print(f"Processing {len(samples)} original samples with {args.max_workers} workers...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        processed = list(tqdm(executor.map(process_fn, samples), total=len(samples)))
        all_data = [res for res in processed if res is not None]

    print(f"Successfully processed {len(all_data)} original samples")

    # Load and process rotation samples if requested
    rotation_processed = []
    if args.rotation_number > 0:
        if args.thinking_mode == "no_thinking":
            print("[WARNING] Rotation QA is only supported in Parquet thinking modes. Skipping rotation.")
        else:
            print(f"Loading rotation samples from {args.rotation_dir}")
            rotation_pairs = load_rotation_samples(args.rotation_dir)
            print(f"Found {len(rotation_pairs)} rotation question-pairs")

            # Filter out test set scenes
            if test_scene_ids:
                before = len(rotation_pairs)
                rotation_pairs = [
                    (d, q) for (d, q) in rotation_pairs
                    if os.path.basename(os.path.dirname(d)) not in test_scene_ids
                ]
                print(f"After filtering test scenes: {len(rotation_pairs)} rotation pairs ({before - len(rotation_pairs)} removed)")

            print(f"Processing {len(rotation_pairs)} rotation samples with {args.max_workers} workers...")
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                rot_processed = list(tqdm(executor.map(process_item_rotation, rotation_pairs), total=len(rotation_pairs)))
            rotation_processed = [r for r in rot_processed if r is not None]
            print(f"Successfully processed {len(rotation_processed)} rotation samples")

    # Remove sample_id before writing
    for item in all_data:
        item.pop('sample_id', None)
    for item in rotation_processed:
        item.pop('sample_id', None)

    # Apply sampling and merge
    combined = apply_sampling(all_data, rotation_processed, args.total_samples, args.rotation_number)
    print(f"Total samples for output: {len(combined)}")

    if len(combined) == 0:
        print("No samples to write. Exiting.")
        return

    os.makedirs(output_dir, exist_ok=True)

    if args.thinking_mode == "no_thinking":
        jsonl_file = os.path.join(output_dir, "no_thinking.jsonl")
        print(f"Writing JSONL to {jsonl_file}")
        with open(jsonl_file, 'w') as f:
            for item in combined:
                f.write(json.dumps(item) + '\n')
        print(f"\nDone! Created {jsonl_file} ({len(combined)} samples)")
        sample = combined[0]
        print("\n--- Sample entry ---")
        print(f"ID: {sample['id']}")
        print(f"Images: {sample['image']}")
        print(f"Human: {sample['conversations'][0]['value'][:200]}...")
        print(f"GPT: {sample['conversations'][1]['value']}")
    else:
        schema = pa.schema([
            pa.field("image_list", pa.list_(pa.binary())),
            pa.field("instruction_list", pa.list_(pa.string())),
            pa.field("output_text_list", pa.list_(pa.string())),
        ])

        rows_per_file = args.rows_per_group * args.groups_per_file
        file_index = 0
        parquet_info = {}

        print(f"Writing parquet files to {output_dir}")
        for i in tqdm(range(0, len(combined), rows_per_file)):
            file_data = combined[i:i + rows_per_file]
            parquet_file = os.path.join(output_dir, f"chunk_{file_index}.parquet")
            file_index += 1

            num_row_groups = 0
            with pq.ParquetWriter(parquet_file, schema=schema, version="2.6") as writer:
                for j in range(0, len(file_data), args.rows_per_group):
                    group_data = file_data[j:j + args.rows_per_group]
                    writer.write_table(pa.Table.from_pylist(group_data, schema=schema))
                    num_row_groups += 1

            parquet_info[parquet_file] = {"num_row_groups": num_row_groups, "num_rows": len(file_data)}

        parquet_info_path = os.path.join(output_dir, "parquet_info.json")
        with open(parquet_info_path, 'w') as f:
            json.dump(parquet_info, f, indent=2)

        print(f"\nDone! Created {file_index} parquet file(s) in {output_dir} ({len(combined)} samples)")
        print(f"Parquet info saved to {parquet_info_path}")


if __name__ == "__main__":
    main()
