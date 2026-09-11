#!/usr/bin/env python3
"""
Run inference on Spatial Collaboration MAP dataset using Google Gemini API.
Given two room perspective images and a top-down map, the model determines
whether the map correctly represents the room.
Optionally includes a panorama image as additional context (--use_panorama).
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


def extract_answer(model_output):
    """
    Extracts the answer (A=Yes or B=No) from model generation.
    Priority:
    1. "Final Answer: X"
    2. "(X)"
    3. Last standalone capital letter A or B
    """
    text = model_output.replace("**", "").strip()

    strict_match = re.search(
        r"(?:Final\s+Answer|Answer):\s*([A-B])", text, re.IGNORECASE
    )
    if strict_match:
        return strict_match.group(1).upper()

    bracket_match = re.search(r"[\(\[]([A-B])[\)\]]", text)
    if bracket_match:
        return bracket_match.group(1).upper()

    last_letter_match = re.findall(r"\b([A-B])\b", text)
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


class GeminiMapInference:
    """Run inference on Map Verification task using Gemini API."""

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
        user_1_image_bytes: bytes,
        user_2_image_bytes: bytes,
        map_image_bytes: bytes,
        question: str,
        options: List[str],
        correct_answer_idx: int,
        correct_answer: str,
        question_type: str,
        panorama_image_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Run inference on a single sample with two perspective images, optionally a panorama, and a map."""

        options_str = "\n".join(
            [f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)]
        )
        full_question = f"\nQUESTION:{question}\n\nOPTIONS:\n{options_str}"

        if panorama_image_bytes is not None:
            system_prompt = (
                "1. Context: You are provided with four images related to the same room.\n"
                "   - Image 1 and Image 2 show two different viewpoints of the room.\n"
                "   - Image 3 is a panoramic view of the room, covering the combined field of view of both viewpoints.\n"
                "   - Image 4 is a top-down map of the room.\n"
                "2. Task: By comparing the spatial layout, object positions, and relationships visible in "
                "the two viewpoint images and the panoramic view against the top-down map, determine whether "
                "the map accurately represents the room.\n"
                "3. Format: First, provide a brief step-by-step reasoning process. Then, identify the single correct option.\n"
                "4. Output: End your response with 'Final Answer: [Option Letter]'.\n"
            )
        else:
            system_prompt = (
                "1. Context: You are provided with two images showing different perspectives of the same room, "
                "and a third image showing a top-down map of the room.\n"
                "2. Task: By comparing the spatial layout, object positions, and relationships visible in "
                "the two perspective images against the top-down map, determine whether the map accurately "
                "represents the room.\n"
                "3. Format: First, provide a brief step-by-step reasoning process. Then, identify the single correct option.\n"
                "4. Output: End your response with 'Final Answer: [Option Letter]'.\n"
            )

        image1_part = types.Part.from_bytes(
            data=user_1_image_bytes,
            mime_type="image/png"
        )
        image2_part = types.Part.from_bytes(
            data=user_2_image_bytes,
            mime_type="image/png"
        )

        full_prompt = system_prompt + "\n" + full_question

        if panorama_image_bytes is not None:
            panorama_part = types.Part.from_bytes(
                data=panorama_image_bytes,
                mime_type="image/png"
            )
            map_part = types.Part.from_bytes(
                data=map_image_bytes,
                mime_type="image/png"
            )
            contents = [image1_part, image2_part, panorama_part, map_part, full_prompt]
        else:
            map_part = types.Part.from_bytes(
                data=map_image_bytes,
                mime_type="image/png"
            )
            contents = [image1_part, image2_part, map_part, full_prompt]

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level)
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


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on Map Verification dataset using Gemini API",
        epilog="Note: Set VisualCoT_GEMINI environment variable.",
    )
    parser.add_argument(
        "--data_file",
        default="/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_dataset_map_questions_normalized.json",
    )
    parser.add_argument(
        "--model_name",
        default="gemini-3-pro-preview",
        help="Gemini model name (e.g., gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--thinking_level",
        default="high",
        choices=["low", "high"],
        help="Thinking level for Gemini (default: high)",
    )
    parser.add_argument("--output_file", default="/path/to/scratch/spatial_collab/Gemini/inference_results_gemini_map.json")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=16384,
                        help="Max output tokens (default: 16384, needs to be high when thinking is enabled)")
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
    parser.add_argument(
        "--use_panorama",
        action="store_true",
        help="Include panorama image as additional input (Image 3=panorama, Image 4=map). "
             "Samples without panorama_path will be skipped.",
    )

    args = parser.parse_args()

    print(f"Loading data from {args.data_file}")
    with open(args.data_file, "r") as f:
        all_data = json.load(f)
    print(f"Total samples loaded: {len(all_data)}")

    if args.use_panorama:
        print("Panorama mode enabled: filtering samples with panorama_path...")
        valid_data = []
        missing_panorama = 0
        for sample in all_data:
            panorama_path = remap_path(sample.get("panorama_path"))
            if panorama_path and os.path.exists(panorama_path):
                sample["_panorama_path"] = panorama_path
                valid_data.append(sample)
            else:
                missing_panorama += 1
        print(f"Samples with panorama: {len(valid_data)}, missing: {missing_panorama}")
        all_data = valid_data

    if args.num_samples is not None and args.num_samples < len(all_data):
        random.seed(args.random_seed)
        all_data = random.sample(all_data, args.num_samples)
        print(f"Randomly sampled {args.num_samples} samples (seed={args.random_seed})")

    # Initialize checkpoint manager
    checkpoint_path = args.output_file.replace(".json", "_checkpoint.json")
    checkpoint_mgr = CheckpointManager(checkpoint_path, save_interval=args.checkpoint_interval)

    # Auto-detect and load existing checkpoint or output file
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
        print(f"Ignoring existing checkpoint/output file (--no_resume flag set)")

    inference_engine = GeminiMapInference(
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking_level=args.thinking_level,
    )

    def load_images(example: Dict[str, Any]):
        """Load images: two perspectives + optionally panorama + map."""
        user_1_image_path = remap_path(example["user_1_image_local_path"])
        user_2_image_path = remap_path(example["user_2_image_local_path"])
        map_image_path = remap_path(example["map_image_path"])

        question = example["question_both_views"]
        options = example["options_user_2"]
        correct_answer_idx = example["user_2_gt_answer_idx"]
        correct_answer = example["user_2_gt_answer_text"]

        with open(user_1_image_path, "rb") as f:
            user_1_image_bytes = f.read()
        with open(user_2_image_path, "rb") as f:
            user_2_image_bytes = f.read()
        with open(map_image_path, "rb") as f:
            map_image_bytes = f.read()

        panorama_image_bytes = None
        if args.use_panorama:
            panorama_path = example["_panorama_path"]
            with open(panorama_path, "rb") as f:
                panorama_image_bytes = f.read()

        question_type = example.get("question_type", "")

        return {
            "user_1_image_bytes": user_1_image_bytes,
            "user_2_image_bytes": user_2_image_bytes,
            "map_image_bytes": map_image_bytes,
            "panorama_image_bytes": panorama_image_bytes,
            "question": question,
            "options": options,
            "correct_answer_idx": correct_answer_idx,
            "correct_answer": correct_answer,
            "question_type": question_type,
            "sample_id": example.get("sample_id", ""),
        }

    def process_single_sample(example: Dict[str, Any]):
        sample_id = example.get("sample_id", "")
        try:
            loaded_data = load_images(example)

            result = inference_engine.run_single_sample(
                user_1_image_bytes=loaded_data["user_1_image_bytes"],
                user_2_image_bytes=loaded_data["user_2_image_bytes"],
                map_image_bytes=loaded_data["map_image_bytes"],
                question=loaded_data["question"],
                options=loaded_data["options"],
                correct_answer_idx=loaded_data["correct_answer_idx"],
                correct_answer=loaded_data["correct_answer"],
                question_type=loaded_data["question_type"],
                panorama_image_bytes=loaded_data["panorama_image_bytes"],
            )

            result["sample_id"] = loaded_data["sample_id"]
            result["question_type"] = loaded_data["question_type"]
            result["question"] = loaded_data["question"]
            result["options"] = loaded_data["options"]
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
        print(f"Checkpoint saved. Will auto-resume on next run.")
        return

    # Save final checkpoint
    checkpoint_mgr.save_final()

    # Get all results
    results = checkpoint_mgr.get_all_results()

    total_accuracy = sum(r["accuracy"] for r in results)
    final_accuracy = total_accuracy / len(results) if results else 0.0

    # Per-answer breakdown (Yes vs No)
    yes_results = [r for r in results if r["correct_answer"] == "Yes"]
    no_results = [r for r in results if r["correct_answer"] == "No"]
    yes_acc = sum(r["accuracy"] for r in yes_results) / len(yes_results) if yes_results else 0.0
    no_acc = sum(r["accuracy"] for r in no_results) / len(no_results) if no_results else 0.0

    print(f"\n{'='*60}")
    print(f"Inference Complete!")
    print(f"Total Samples: {len(results)}")
    print(f"Overall Accuracy: {final_accuracy:.4f} ({int(total_accuracy)}/{len(results)})")
    print(f"  Yes (correct map) Accuracy: {yes_acc:.4f} ({int(sum(r['accuracy'] for r in yes_results))}/{len(yes_results)})")
    print(f"  No (incorrect map) Accuracy: {no_acc:.4f} ({int(sum(r['accuracy'] for r in no_results))}/{len(no_results)})")
    print(f"{'='*60}")

    output_data = {
        "config": {
            "model_name": args.model_name,
            "thinking_level": args.thinking_level,
            "num_samples": len(results),
            "temperature": args.temperature,
            "use_panorama": args.use_panorama,
            "num_images": 4 if args.use_panorama else 3,
        },
        "metrics": {
            "overall_accuracy": final_accuracy,
            "total_correct": total_accuracy,
            "total_samples": len(results),
            "yes_accuracy": yes_acc,
            "no_accuracy": no_acc,
        },
        "results": results,
    }

    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output_file}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint file removed: {checkpoint_path}")


if __name__ == "__main__":
    main()
