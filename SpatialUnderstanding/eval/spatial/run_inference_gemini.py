#!/usr/bin/env python3
"""
Run inference on Spatial Collaboration dataset using Google Gemini API.
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
    Extracts the multiple-choice letter (A, B, C, or D) from model generation.
    Priority:
    1. "Final Answer: X"
    2. "(X)"
    3. Last standalone capital letter
    """
    # 1. Clean the text: remove bolding and trailing whitespace
    text = model_output.replace("**", "").strip()

    # 2. Pattern: Final Answer: A or Answer: A
    strict_match = re.search(
        r"(?:Final\s+Answer|Answer):\s*([A-D])", text, re.IGNORECASE
    )
    if strict_match:
        return strict_match.group(1).upper()

    # 3. Pattern: (A) or [A]
    bracket_match = re.search(r"[\(\[]([A-D])[\)\]]", text)
    if bracket_match:
        return bracket_match.group(1).upper()

    # 4. Pattern: The last standalone capital letter at the end of a sentence or string
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
        self.results = {}  # sample_id -> result
        self.completed_count = 0

    def load_checkpoint(self) -> Dict[str, Dict]:
        """Load existing checkpoint if available, filtering out empty responses."""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r") as f:
                    data = json.load(f)
                    all_results = data.get("results", [])

                    # Filter out results with empty final_answer_text (need to recompute)
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
        """Return set of already completed sample_ids (with non-empty responses)."""
        return set(self.results.keys())

    def add_result(self, result: Dict):
        """Add a result and save checkpoint periodically."""
        with self.lock:
            sample_id = result["sample_id"]
            self.results[sample_id] = result
            self.completed_count += 1

            # Save checkpoint every N samples
            if self.completed_count % self.save_interval == 0:
                self._save_checkpoint()

    def _save_checkpoint(self):
        """Save current results to checkpoint file."""
        checkpoint_data = {
            "results": list(self.results.values()),
            "total_completed": len(self.results),
        }
        # Write to temp file first, then rename for atomicity
        temp_path = self.checkpoint_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(checkpoint_data, f)
        os.replace(temp_path, self.checkpoint_path)

    def save_final(self):
        """Save final checkpoint."""
        with self.lock:
            self._save_checkpoint()
            print(f"Checkpoint saved: {len(self.results)} samples")

    def get_all_results(self) -> List[Dict]:
        """Get all results as a list."""
        with self.lock:
            return list(self.results.values())


class GeminiSpatialCollabInference:
    """Run inference on Spatial Collaboration task using Gemini API."""

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

        # Initialize Gemini client
        # Requires VisualCoT_GEMINI or GEMINI_API_KEY environment variable
        api_key = os.environ.get("VisualCoT_GEMINI")
        if not api_key:
            raise ValueError(
                "VisualCoT_GEMINI environment variable is required"
            )

        self.client = genai.Client(api_key=api_key)
        print(f"Initialized Gemini client with model: {model_name}")

    def run_single_sample(
        self,
        answerer_image_bytes: bytes,
        helper_image_bytes: bytes,
        question: str,
        options: List[str],
        correct_answer_idx: int,
        correct_answer: str,
        question_type: str,
    ) -> Dict[str, Any]:
        """Run inference on a single sample."""

        options_str = "\n".join(
            [f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)]
        )
        full_question = f"\nQUESTION:{question}\n\nOPTIONS:{options_str}"

        system_prompt = (
            "1. Context: You are provided with two images showing different perspectives of the same room.\n"
            "2. Task: Analyze the spatial relationships and objects in both views to answer the following multiple-choice question.\n"
            "3. Format: First, provide a brief step-by-step reasoning process. Then, identify the single correct option.\n"
            "4. Output: End your response with 'Final Answer: [Option Letter]'.\n"
        )

        # Create image parts for Gemini
        image1_part = types.Part.from_bytes(
            data=answerer_image_bytes,
            mime_type="image/png"
        )
        image2_part = types.Part.from_bytes(
            data=helper_image_bytes,
            mime_type="image/png"
        )

        # Combine system prompt with user content
        full_prompt = system_prompt + "\n" + full_question

        # Build content with images and text
        contents = [image1_part, image2_part, full_prompt]

        # Configure generation
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

        # Handle cases where response text is None (e.g., MAX_TOKENS reached during thinking)
        if response.text is None:
            finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
            thoughts_tokens = getattr(response.usage_metadata, 'thoughts_token_count', 0)
            print(f"Warning: Empty response (finish_reason={finish_reason}, thoughts_tokens={thoughts_tokens})")
            answerer_response = ""
        else:
            answerer_response = response.text.strip()

        predicted_answer = extract_answer(answerer_response)
        accuracy = calculate_accuracy(predicted_answer, correct_answer_idx)

        return {
            "final_answer_text": answerer_response,
            "predicted_answer": predicted_answer,
            "correct_answer": correct_answer,
            "correct_answer_idx": correct_answer_idx,
            "accuracy": accuracy,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on Spatial Collaboration dataset using Gemini API",
        epilog="Note: Set VisualCoT_GEMINI or GEMINI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--data_dir", default="/path/to/scratch/spatial_collab_dataset"
    )
    parser.add_argument(
        "--dataset_files", nargs="+", default=["anchor_dataset_V_Final_2000.json"]
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
    parser.add_argument("--output_file", default="/path/to/scratch/spatial_collab/Gemini/inference_results_gemini.json")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=16384,
                        help="Max output tokens (default: 8192, needs to be high when thinking is enabled)")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Maximum number of parallel workers for inference (default: 32)",
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

    all_data = []
    for dataset_file in args.dataset_files:
        file_path = os.path.join(args.data_dir, dataset_file)
        print(f"Loading data from {file_path}")
        with open(file_path, "r") as f:
            data = json.load(f)
            all_data.extend(data)
            print(f"Loaded {len(data)} samples from {dataset_file}")

    print(f"Total samples loaded: {len(all_data)}")

    if args.num_samples is not None and args.num_samples < len(all_data):
        random.seed(args.random_seed)
        all_data = random.sample(all_data, args.num_samples)
        print(f"Randomly sampled {args.num_samples} samples (seed={args.random_seed})")

    # Initialize checkpoint manager
    checkpoint_path = args.output_file.replace(".json", "_checkpoint.json")
    checkpoint_mgr = CheckpointManager(checkpoint_path, save_interval=args.checkpoint_interval)

    # Auto-detect and load existing checkpoint or output file (unless --no_resume is set)
    completed_sample_ids = set()
    if not args.no_resume:
        if os.path.exists(checkpoint_path):
            # First priority: load from checkpoint file
            print(f"Checkpoint detected: {checkpoint_path}")
            checkpoint_mgr.load_checkpoint()
            completed_sample_ids = checkpoint_mgr.get_completed_sample_ids()
            print(f"Auto-resuming from checkpoint: {len(completed_sample_ids)} samples already completed, {len(all_data) - len(completed_sample_ids)} remaining")
        elif os.path.exists(args.output_file):
            # Second priority: load from output file if no checkpoint exists
            print(f"No checkpoint found, but output file exists: {args.output_file}")
            try:
                with open(args.output_file, "r") as f:
                    output_data = json.load(f)
                    results_from_output = output_data.get("results", [])

                    # Filter out results with empty final_answer_text
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

    inference_engine = GeminiSpatialCollabInference(
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking_level=args.thinking_level,
    )

    def load_images(example: Dict[str, Any]):
        """Load images synchronously in thread pool."""
        user_1_image_path = example["user_1_image_local_path"].replace(
            "/path/to/scratch", "/path/to/scratch"
        )
        user_2_image_path = example["user_2_image_local_path"].replace(
            "/path/to/scratch", "/path/to/scratch"
        )
        question = example["question_both_views"].replace(
            "both your and your partner's perspectives", "both images"
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
            answerer_image_bytes = f.read()
        with open(user_2_image_path, "rb") as f:
            helper_image_bytes = f.read()

        question_type = example.get("question_type", "")

        return {
            "answerer_image_bytes": answerer_image_bytes,
            "helper_image_bytes": helper_image_bytes,
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
                answerer_image_bytes=loaded_data["answerer_image_bytes"],
                helper_image_bytes=loaded_data["helper_image_bytes"],
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
            return result
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Filter out already completed samples using sample_id
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

                    # Show intermediate accuracy
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
