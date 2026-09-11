#!/usr/bin/env python3
"""
Regenerate ONLY relative_distance questions for all accepted scenes,
patch into existing questions_paraphrased.json, then re-run aggregate_data + filter_questions.

This avoids re-running all other expensive pipeline stages (paraphrase for other question types,
generate_descriptions, solve_perception, generate_maps, etc.).

Usage:
    python regen_relative_distance.py \
        --base_dir /path/to/scratch/infinigen/spatial \
        --question_version V5 \
        --api_key /path/to/key_file \
        [--dist_threshold 0.5]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Add repo root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

QUESTION_GENERATION_DIR = REPO_ROOT / "MultiAgent_Spatial" / "question_generation_v2"
GENERATE_QUESTIONS_SCRIPT = QUESTION_GENERATION_DIR / "generate_questions.py"
DATAGEN_PIPELINE_SCRIPT = SCRIPT_DIR / "datagen_pipeline.py"


def find_accepted_scenes(base_dir: Path) -> List[Path]:
    scenes = []
    for accept_file in base_dir.rglob("ACCEPT.txt"):
        scene_dir = accept_file.parent
        scenes.append(scene_dir)
    return sorted(scenes)


def generate_relative_distance_questions(scene: Path, dist_threshold: float) -> Optional[List]:
    """Run generate_questions.py for a scene, return only relative_distance_questions."""
    required = [
        scene / "visible_objects_with_descriptions.json",
        scene / "visible_objects.json",
        scene / "cameras.json",
    ]
    for f in required:
        if not f.exists():
            print(f"  [SKIP] Missing {f.name} in {scene.name}")
            return None

    output_json = scene / "questions_reldist_new.json"
    cmd = [
        sys.executable, str(GENERATE_QUESTIONS_SCRIPT),
        "--input_json", str(scene / "visible_objects_with_descriptions.json"),
        "--ground_truth_json", str(scene / "visible_objects.json"),
        "--cam_data_file", str(scene / "cameras.json"),
        "--dist_threshold", str(dist_threshold),
        "--output_json", str(output_json),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] generate_questions.py failed for {scene.name}: {result.stderr[-200:]}")
        return None

    if not output_json.exists():
        print(f"  [ERROR] Output not produced for {scene.name}")
        return None

    with open(output_json) as f:
        data = json.load(f)

    output_json.unlink()  # Clean up temp file
    return data.get("relative_distance_questions", [])


def paraphrase_relative_distance_questions(
    scenes_and_questions: List[Tuple[Path, List]],
    model_name: str,
    api_key: str,
) -> Dict[Path, List]:
    """Paraphrase relative_distance_questions for all scenes using native google.genai SDK."""
    from question_generation_v2.paraphrase_questions import build_paraphrase_prompt, extract_json_from_output
    from google import genai
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = genai.Client(api_key=api_key)

    all_queries = []
    query_map = []  # (scene, idx, original_q)

    for scene, questions in scenes_and_questions:
        for idx, q in enumerate(questions):
            try:
                prompt, json_to_paraphrase = build_paraphrase_prompt(q, "relative_distance_questions")
            except Exception as e:
                print(f"  [WARN] Failed to build prompt for {scene.name}[{idx}]: {e}")
                continue
            full_prompt = prompt + f"\n{json.dumps(json_to_paraphrase, indent=2)}"
            all_queries.append(full_prompt)
            query_map.append((scene, idx, q))

    if not all_queries:
        print("No paraphrase queries to send.")
        return {}

    print(f"Sending {len(all_queries)} paraphrase queries...")

    def call_gemini(prompt_text: str) -> str:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt_text,
            )
            return response.text or ""
        except Exception as e:
            print(f"  [WARN] Gemini API error: {e}")
            return ""

    outputs = [None] * len(all_queries)
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_idx = {executor.submit(call_gemini, q): i for i, q in enumerate(all_queries)}
        for future in tqdm(as_completed(future_to_idx), total=len(all_queries), desc="Paraphrasing"):
            i = future_to_idx[future]
            outputs[i] = future.result()

    scene_results: Dict[Path, List] = {}
    for text, (scene, idx, original_q) in zip(outputs, query_map):
        paraphrased = extract_json_from_output(text or "")
        if not paraphrased:
            print(f"  [WARN] Empty paraphrase response for {scene.name}[{idx}]")
            continue

        final_q = original_q.copy()
        final_q.update(paraphrased)

        # Validate options
        options = final_q.get("options")
        if not isinstance(options, list) or len(options) < 2:
            print(f"  [WARN] Invalid options for {scene.name}[{idx}], skipping")
            continue

        # Set correct_answer
        correct_index = final_q.get("correct_index")
        if correct_index is None or not (0 <= correct_index < len(options)):
            print(f"  [WARN] Invalid correct_index for {scene.name}[{idx}], skipping")
            continue
        final_q["correct_answer"] = options[correct_index]

        scene_results.setdefault(scene, []).append(final_q)

    return scene_results


def patch_paraphrased_json(scene: Path, new_rel_dist_questions: list) -> bool:
    """Replace relative_distance_questions in existing questions_paraphrased.json.
    If the file doesn't exist, create it from scratch with empty other question types,
    and also create a stub questions.json so aggregate_data's existence check passes.
    """
    paraphrase_path = scene / "questions_paraphrased.json"
    questions_path = scene / "questions.json"

    if not paraphrase_path.exists():
        # Create from scratch — scene had no prior QA pipeline output
        data = {
            "counting_questions": [],
            "anchor_questions": [],
            "spatial_orientation_questions": [],
            "perspective_taking_questions": [],
            "relative_distance_questions": [],
        }
        # Also create stub questions.json so aggregate_data existence check passes
        if not questions_path.exists():
            stub = {
                "counting_questions": [],
                "anchor_questions": [],
                "spatial_orientation_questions": [],
                "perspective_taking_questions": [],
                "relative_distance_questions": [],
            }
            with open(questions_path, "w") as f:
                json.dump(stub, f, indent=2)
            print(f"  {scene.name}: created stub questions.json")
        old_count = 0
    else:
        with open(paraphrase_path) as f:
            data = json.load(f)
        old_count = len(data.get("relative_distance_questions", []))

    data["relative_distance_questions"] = new_rel_dist_questions
    new_count = len(new_rel_dist_questions)

    with open(paraphrase_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  {scene.name}: relative_distance {old_count} → {new_count}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True, help="Base dir (e.g. .../spatial or .../outputs_rendered)")
    parser.add_argument("--question_version", default="V5", help="V4 or V5")
    parser.add_argument("--scene_datafile", default=None, help="Path to scene datafile JSON (passed to datagen_pipeline)")
    parser.add_argument("--dist_threshold", type=float, default=0.5)
    parser.add_argument("--model_name_paraphrase", default="gemini-3-flash-preview")
    parser.add_argument("--api_key", default=None, help="Path to API key file or key string")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    sys.path.insert(0, str(QUESTION_GENERATION_DIR.parent))

    # Resolve API key
    api_key = args.api_key
    if api_key and Path(api_key).exists():
        api_key = Path(api_key).read_text().strip()
    if not api_key:
        api_key = os.environ.get("VisualCoT_GEMINI", "")

    # Step 1: Find all accepted scenes
    scenes = find_accepted_scenes(base_dir)
    print(f"Found {len(scenes)} accepted scenes in {base_dir}")

    # Step 2: Generate new relative_distance questions for each scene
    print(f"\n=== Step 1: Generating relative_distance questions (dist_threshold={args.dist_threshold}) ===")
    scenes_and_questions = []
    total_new_raw = 0
    for scene in tqdm(scenes, desc="Generating"):
        new_qs = generate_relative_distance_questions(scene, args.dist_threshold)
        if new_qs is None:
            continue
        scenes_and_questions.append((scene, new_qs))
        total_new_raw += len(new_qs)
        if new_qs:
            print(f"  {scene.name}: {len(new_qs)} raw relative_distance questions")

    print(f"\nTotal raw relative_distance questions: {total_new_raw} from {len(scenes_and_questions)} scenes")

    # Step 3: Paraphrase
    print(f"\n=== Step 2: Paraphrasing relative_distance questions ===")
    scene_paraphrased = paraphrase_relative_distance_questions(
        scenes_and_questions,
        model_name=args.model_name_paraphrase,
        api_key=api_key,
    )

    # Step 4: Patch questions_paraphrased.json
    print(f"\n=== Step 3: Patching questions_paraphrased.json ===")
    for scene, paraphrased_qs in scene_paraphrased.items():
        patch_paraphrased_json(scene, paraphrased_qs)
    # Scenes with 0 new questions: patch with empty list
    for scene, raw_qs in scenes_and_questions:
        if scene not in scene_paraphrased:
            paraphrase_path = scene / "questions_paraphrased.json"
            if paraphrase_path.exists():
                patch_paraphrased_json(scene, [])

    # Step 5: Re-run aggregate_data + filter_questions via datagen_pipeline
    print(f"\n=== Step 4: Re-running aggregate_data + filter_questions ===")
    scene_datafile = args.scene_datafile
    if scene_datafile is None:
        # Default based on question version
        scene_datafile = str(SCRIPT_DIR / ("dataset_new_scenes_v5.json" if args.question_version == "V5" else "dataset_new_scenes_v1.json"))

    # Write api_key to temp file so datagen_pipeline can read it
    import tempfile
    tmp_key_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(api_key if api_key else "")
            tmp_key_path = f.name

        cmd = [
            sys.executable, str(DATAGEN_PIPELINE_SCRIPT),
            "--base_dir", str(base_dir),
            "--scene_datafile", scene_datafile,
            "--stages_to_run", "aggregate_data", "filter_questions",
            "--question_version", args.question_version,
            "--api_key", tmp_key_path,
            "--overwrite_files",
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[ERROR] datagen_pipeline aggregate+filter failed (exit {result.returncode})")
            sys.exit(1)
    finally:
        if tmp_key_path and os.path.exists(tmp_key_path):
            os.unlink(tmp_key_path)

    print(f"\nDone. Check dataset_relative_distance_questions_filtered_{args.question_version}.json")


if __name__ == "__main__":
    main()
