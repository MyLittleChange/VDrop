#!/usr/bin/env python3
"""
Run inference on Spatial Collaboration dataset using Google Gemini API with
two perspective images + a top-down map.
Supports checkpointing to resume from interruptions.
"""

import argparse
import json
import os
import re
import random
import threading
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types


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


def extract_answer(model_output):
    """
    Extracts the multiple-choice letter (A, B, C, or D) from model generation.
    Priority:
    1. "Final Answer: X"
    2. "(X)"
    3. Last standalone capital letter
    """
    text = model_output.replace("**", "").strip()

    strict_match = re.search(
        r"(?:Final\s+Answer|Answer):\s*([A-D])", text, re.IGNORECASE
    )
    if strict_match:
        return strict_match.group(1).upper()

    bracket_match = re.search(r"[\(\[]([A-D])[\)\]]", text)
    if bracket_match:
        return bracket_match.group(1).upper()

    last_letter_match = re.findall(r"\b([A-D])\b", text)
    if last_letter_match:
        return last_letter_match[-1].upper()

    return None


def calculate_accuracy(answer: Optional[str], correct_answer_idx: int) -> float:
    """Calculate if the answer is correct."""
    if answer is None:
        return 0.0
    expected_letter = chr(65 + correct_answer_idx)
    return 1.0 if answer == expected_letter else 0.0


class CheckpointManager:
    """Manages checkpointing for resumable inference using sample_id as unique key."""

    def __init__(self, checkpoint_path: str, save_interval: int = 10):
        self.checkpoint_path = checkpoint_path
        self.save_interval = save_interval
        self.lock = threading.Lock()
        self.results = {}
        self.completed_count = 0

    def load_checkpoint(self) -> Dict[str, Dict]:
        """Load existing checkpoint if available, filtering out empty responses."""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r") as f:
                    data = json.load(f)
                    all_results = data.get("results", [])

                    valid_results = []
                    empty_count = 0
                    for r in all_results:
                        if r.get("final_answer_text", "").strip():
                            valid_results.append(r)
                        else:
                            empty_count += 1

                    self.results = {r["sample_id"]: r for r in valid_results}
                    print(f"Loaded checkpoint with {len(self.results)} completed samples")
                    if empty_count > 0:
                        print(f"Filtered out {empty_count} samples with empty responses (will recompute)")
                    return self.results
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load checkpoint: {e}")
        return {}

    def get_completed_sample_ids(self) -> set:
        return set(self.results.keys())

    def add_result(self, result: Dict):
        with self.lock:
            sample_id = result["sample_id"]
            self.results[sample_id] = result
            self.completed_count += 1
            if self.completed_count % self.save_interval == 0:
                self._save_checkpoint()

    def _save_checkpoint(self):
        checkpoint_data = {
            "results": list(self.results.values()),
            "total_completed": len(self.results),
        }
        temp_path = self.checkpoint_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(checkpoint_data, f)
        os.replace(temp_path, self.checkpoint_path)

    def save_final(self):
        with self.lock:
            self._save_checkpoint()
            print(f"Checkpoint saved: {len(self.results)} samples")

    def get_all_results(self) -> List[Dict]:
        with self.lock:
            return list(self.results.values())


class GeminiThreeImageInference:
    """Run inference on Spatial Collaboration task using Gemini API with three images."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro-preview-06-05",
        temperature: float = 0.7,
        max_tokens: int = 512,
        thinking_level: str = "low",
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.thinking_level = thinking_level

        api_key = os.environ.get("VisualCoT_GEMINI")
        if not api_key:
            raise ValueError(
                "VisualCoT_GEMINI environment variable is required"
            )

        self.client = genai.Client(api_key=api_key)
        print(f"Initialized Gemini client with model: {model_name}")

    def run_single_sample(
        self,
        image1_bytes: bytes,
        image2_bytes: bytes,
        image3_bytes: bytes,
        question: str,
        options: List[str],
        correct_answer_idx: int,
        correct_answer: str,
        question_type: str,
    ) -> Dict[str, Any]:
        """Run inference on a single sample with three images."""

        options_str = "\n".join(
            [f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)]
        )
        full_question = f"\nQUESTION:{question}\n\nOPTIONS:{options_str}"

        system_prompt = (
            "1. Context: You are provided with three images related to the same room.\n"
            "   - Image 1 and Image 2 show two different viewpoints of the room.\n"
            "   - Image 3 is a top-down map of the room from above, oriented to match Image 2's viewpoint.\n"
            "2. Task: Analyze the spatial relationships and objects in the two viewpoint images, "
            "using the top-down map to understand the overall room layout, and answer the following multiple-choice question.\n"
            "3. Format: First, provide a brief step-by-step reasoning process. Then, identify the single correct option.\n"
            "4. Output: End your response with 'Final Answer: [Option Letter]'.\n"
        )

        image1_part = types.Part.from_bytes(
            data=image1_bytes,
            mime_type="image/png"
        )
        image2_part = types.Part.from_bytes(
            data=image2_bytes,
            mime_type="image/png"
        )
        image3_part = types.Part.from_bytes(
            data=image3_bytes,
            mime_type="image/png"
        )

        full_prompt = system_prompt + "\n" + full_question
        contents = [image1_part, image2_part, image3_part, full_prompt]

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_level=self.thinking_level,
                include_thoughts=True,
            )
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        if response.text is None:
            finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
            thoughts_tokens = getattr(response.usage_metadata, 'thoughts_token_count', 0)
            print(f"Warning: Empty response (finish_reason={finish_reason}, thoughts_tokens={thoughts_tokens})")
            model_response = ""
        else:
            model_response = response.text.strip()

        predicted_answer = extract_answer(model_response)
        accuracy = calculate_accuracy(predicted_answer, correct_answer_idx)

        return {
            "final_answer_text": model_response,
            "predicted_answer": predicted_answer,
            "correct_answer": correct_answer,
            "correct_answer_idx": correct_answer_idx,
            "accuracy": accuracy,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on Spatial Collaboration dataset using Gemini API with 2 images + top-down map",
        epilog="Note: Set VisualCoT_GEMINI environment variable.",
    )
    parser.add_argument(
        "--dataset_file",
        default="/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
        help="Path to the dataset JSON file (must have topdown_path field)",
    )
    parser.add_argument(
        "--model_name",
        default="gemini-3-pro-preview",
        help="Gemini model name",
    )
    parser.add_argument(
        "--thinking_level",
        default="high",
        choices=["low", "high"],
        help="Thinking level for Gemini (default: high)",
    )
    parser.add_argument(
        "--output_file",
        default="/path/to/scratch/spatial_collab/Gemini/inference_results_gemini_topdown.json",
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--max_tokens", type=int, default=16384,
        help="Max output tokens (default: 16384, needs to be high when thinking is enabled)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Maximum number of parallel workers for inference (default: 1)",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="Save checkpoint every N samples (default: 10)",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Start fresh, ignoring any existing checkpoint",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset_file}...")
    with open(args.dataset_file, "r") as f:
        all_data = json.load(f)
    print(f"Total samples loaded: {len(all_data)}")

    # Filter to samples that have a topdown_path
    valid_data = [s for s in all_data if s.get("topdown_path")]
    missing_topdown = len(all_data) - len(valid_data)
    if missing_topdown > 0:
        print(f"Skipped {missing_topdown} samples without topdown_path")
    all_data = valid_data
    print(f"Samples with topdown_path: {len(all_data)}")

    if args.num_samples is not None and args.num_samples < len(all_data):
        random.seed(args.random_seed)
        all_data = random.sample(all_data, args.num_samples)
        print(f"Randomly sampled {args.num_samples} samples (seed={args.random_seed})")

    # Initialize checkpoint manager
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    checkpoint_path = args.output_file.replace(".json", "_checkpoint.json")
    checkpoint_mgr = CheckpointManager(checkpoint_path, save_interval=args.checkpoint_interval)

    completed_sample_ids = set()
    if not args.no_resume:
        if os.path.exists(checkpoint_path):
            print(f"Checkpoint detected: {checkpoint_path}")
            checkpoint_mgr.load_checkpoint()
            completed_sample_ids = checkpoint_mgr.get_completed_sample_ids()
            print(f"Auto-resuming from checkpoint: {len(completed_sample_ids)} samples already completed, {len(all_data) - len(completed_sample_ids)} remaining")
        elif os.path.exists(args.output_file):
            print(f"No checkpoint found, but output file exists: {args.output_file}")
            try:
                with open(args.output_file, "r") as f:
                    output_data = json.load(f)
                    results_from_output = output_data.get("results", [])

                    valid_count = 0
                    empty_count = 0
                    for r in results_from_output:
                        sample_id = r.get("sample_id", "")
                        if sample_id and r.get("final_answer_text", "").strip():
                            checkpoint_mgr.results[sample_id] = r
                            valid_count += 1
                        elif sample_id:
                            empty_count += 1

                    completed_sample_ids = checkpoint_mgr.get_completed_sample_ids()
                    print(f"Loaded {valid_count} completed samples from output file")
                    if empty_count > 0:
                        print(f"Filtered out {empty_count} samples with empty responses (will recompute)")
                    print(f"Auto-resuming: {len(completed_sample_ids)} samples already completed, {len(all_data) - len(completed_sample_ids)} remaining")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load output file for resume: {e}")
    elif args.no_resume and (os.path.exists(checkpoint_path) or os.path.exists(args.output_file)):
        print("Ignoring existing checkpoint/output file (--no_resume flag set)")

    inference_engine = GeminiThreeImageInference(
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking_level=args.thinking_level,
    )

    def load_images(example: Dict[str, Any]):
        """Load all three images: user_1, user_2, and topdown map."""
        user_1_image_path = remap_path(example["user_1_image_local_path"])
        user_2_image_path = remap_path(example["user_2_image_local_path"])
        topdown_path = remap_path(example["topdown_path"])

        question = example["question_both_views"].replace(
            "both your and your partner's perspectives", "all three images"
        )

        if example.get("options_user_2") is not None:
            options = example["options_user_2"]
            correct_answer_idx = example["user_2_gt_answer_idx"]
            correct_answer = example["user_2_gt_answer_text"]
        else:
            options = example.get("options_user_1", [])
            correct_answer_idx = example.get("user_1_gt_answer_idx", 0)
            correct_answer = example.get("user_1_gt_answer_text", "")

        with open(user_1_image_path, "rb") as f:
            image1_bytes = f.read()
        with open(user_2_image_path, "rb") as f:
            image2_bytes = f.read()
        with open(topdown_path, "rb") as f:
            image3_bytes = f.read()

        question_type = example.get("question_type", "")

        return {
            "image1_bytes": image1_bytes,
            "image2_bytes": image2_bytes,
            "image3_bytes": image3_bytes,
            "question": question,
            "options": options,
            "correct_answer_idx": correct_answer_idx,
            "correct_answer": correct_answer,
            "question_type": question_type,
            "sample_id": example.get("sample_id", ""),
            "scene_id": example.get("scene_id", ""),
            "room_part": example.get("room_part", ""),
            "user_1_image_path": user_1_image_path,
            "user_2_image_path": user_2_image_path,
            "topdown_path": topdown_path,
            "difficulty": example.get("difficulty", ""),
            "description_difficulty": example.get("description_difficulty", ""),
            "angle": example.get("angle"),
            "distance": example.get("distance"),
            "question_object": example.get("question_object", ""),
            "other_agent_angle": example.get("other_agent_angle"),
            "other_agent_distance": example.get("other_agent_distance"),
        }

    def process_single_sample(example: Dict[str, Any]):
        sample_id = example.get("sample_id", "")
        try:
            loaded_data = load_images(example)

            result = inference_engine.run_single_sample(
                image1_bytes=loaded_data["image1_bytes"],
                image2_bytes=loaded_data["image2_bytes"],
                image3_bytes=loaded_data["image3_bytes"],
                question=loaded_data["question"],
                options=loaded_data["options"],
                correct_answer_idx=loaded_data["correct_answer_idx"],
                correct_answer=loaded_data["correct_answer"],
                question_type=loaded_data["question_type"],
            )

            result["sample_id"] = loaded_data["sample_id"]
            result["question_type"] = loaded_data["question_type"]
            result["question"] = loaded_data["question"]
            result["options"] = loaded_data["options"]
            result["scene_id"] = loaded_data["scene_id"]
            result["room_part"] = loaded_data["room_part"]
            result["user_1_image_path"] = loaded_data["user_1_image_path"]
            result["user_2_image_path"] = loaded_data["user_2_image_path"]
            result["topdown_path"] = loaded_data["topdown_path"]
            result["difficulty"] = loaded_data["difficulty"]
            result["description_difficulty"] = loaded_data["description_difficulty"]
            result["angle"] = loaded_data["angle"]
            result["distance"] = loaded_data["distance"]
            result["question_object"] = loaded_data["question_object"]
            result["other_agent_angle"] = loaded_data["other_agent_angle"]
            result["other_agent_distance"] = loaded_data["other_agent_distance"]
            return result
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Filter out already completed samples
    samples_to_process = [
        example for example in all_data
        if example.get("sample_id", "") not in completed_sample_ids
    ]

    print(f"Running inference with {args.max_workers} parallel workers...")
    print(f"Samples to process: {len(samples_to_process)}")

    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(process_single_sample, example): example.get("sample_id", "")
                for example in samples_to_process
            }

            with tqdm(total=len(futures), desc="Running inference") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        checkpoint_mgr.add_result(result)

                    pbar.update(1)

                    all_results = checkpoint_mgr.get_all_results()
                    if len(all_results) > 0 and len(all_results) % 10 == 0:
                        total_acc = sum(r["accuracy"] for r in all_results)
                        current_acc = total_acc / len(all_results)
                        pbar.set_postfix({"accuracy": f"{current_acc:.4f}", "completed": len(all_results)})

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        checkpoint_mgr.save_final()
        print("Checkpoint saved. Will auto-resume on next run.")
        return

    # Save final checkpoint
    checkpoint_mgr.save_final()

    # Get all results (including previously completed ones)
    results = checkpoint_mgr.get_all_results()

    total_accuracy = sum(r["accuracy"] for r in results)
    final_accuracy = total_accuracy / len(results) if results else 0.0

    print(f"\n{'='*60}")
    print(f"Inference Complete!")
    print(f"Total Samples: {len(results)}")
    print(f"Overall Accuracy: {final_accuracy:.4f} ({total_accuracy}/{len(results)})")
    print(f"{'='*60}")

    output_data = {
        "config": {
            "model_name": args.model_name,
            "thinking_level": args.thinking_level,
            "num_samples": len(results),
            "temperature": args.temperature,
            "num_images": 3,
            "third_image": "topdown_agent_matching_second_image",
        },
        "metrics": {
            "overall_accuracy": final_accuracy,
            "total_correct": total_accuracy,
            "total_samples": len(results),
        },
        "results": results,
    }

    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output_file}")

    # Clean up checkpoint file after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint file removed: {checkpoint_path}")


if __name__ == "__main__":
    main()
