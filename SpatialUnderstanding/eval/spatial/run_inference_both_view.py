#!/usr/bin/env python3
"""
Run inference on Spatial Collaboration dataset using vLLM (API mode) for ANSWERER AGENT
and OpenAI API for HELPER AGENT.
"""

import argparse
import json
import os
import re
import base64
import random

# CHANGED: Added List, Dict, Tuple, Any for Python < 3.9 compatibility
from typing import Optional, List, Dict, Tuple, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

def image_to_base64_url(image_bytes: bytes) -> str:
    """Convert image bytes to base64 data URL."""
    img_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


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
    # This is useful if the model says "The correct option is B."
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


class SpatialCollabInference:
    """Run inference on Spatial Collaboration task."""

    def __init__(
        self,
        answerer_api_base: str,
        answerer_model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Determine which models need max_completion_tokens instead of max_tokens
        def uses_max_completion_tokens(model_name: str) -> bool:
            """Check if model uses max_completion_tokens instead of max_tokens."""
            # GPT-5 models and newer use max_completion_tokens
            return (
                "gpt-5" in model_name.lower()
                or "o1" in model_name.lower()
                or "o3" in model_name.lower()
            )

        self.answerer_uses_completion_tokens = uses_max_completion_tokens(
            answerer_model_name
        )

        # Get OpenAI API key from environment
        openai_api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")

        # Determine if we need a real OpenAI API key for each client
        def needs_openai_key(api_base: str) -> bool:
            """Check if the API base is an OpenAI endpoint."""
            if api_base is None:
                return False
            return "openai.com" in api_base.lower() or "api.openai" in api_base.lower()

        # Configure answerer client
        print(f"Connecting to answerer API at {answerer_api_base}...")
        answerer_key = (
            openai_api_key if needs_openai_key(answerer_api_base) else "EMPTY"
        )
        if needs_openai_key(answerer_api_base) and answerer_key == "EMPTY":
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAI API endpoints"
            )

        self.answerer_client = OpenAI(
            api_key=answerer_key,
            base_url=answerer_api_base if answerer_api_base else None,
        )
        self.answerer_model_name = answerer_model_name

    def run_single_sample(
        self,
        answerer_image_bytes: bytes,
        helper_image_bytes: bytes,
        question: str,
        options: List[str],  # CHANGED: list[str] -> List[str]
        correct_answer_idx: int,
        correct_answer: str,
        question_type: str,
    ) -> Dict[str, Any]:  # CHANGED: dict -> Dict[str, Any]
        """Run inference on a single sample."""

        options_str = "\n".join(
            [f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)]
        )
        full_question = f"{question}\n\nOptions:{options_str}"

        ANSWERER_SYSTEM_PROMPT = (
            "Let's think step by step to answer the question. Enclose your thinking process within <think> </think> tags. Finally conclude with the final answer wrapped in <answer></answer> tags."
        )

        answerer_messages = [
            {"role": "system", "content": ANSWERER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_base64_url(answerer_image_bytes)},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_base64_url(helper_image_bytes)},
                    },
                    {"type": "text", "text": full_question},
                ],
            },
        ]

        # Use appropriate token parameter based on model
        if self.answerer_uses_completion_tokens:
            response = self.answerer_client.chat.completions.create(
                model=self.answerer_model_name,
                messages=answerer_messages,
                max_completion_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        else:
            response = self.answerer_client.chat.completions.create(
                model=self.answerer_model_name,
                messages=answerer_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        answerer_response = response.choices[0].message.content.strip()

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
        description="Run inference on Spatial Collaboration dataset",
        epilog="Note: For OpenAI API endpoints, set OPENAI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--data_dir", default="/path/to/scratch/spatial_collab"
    )
    parser.add_argument(
        "--dataset_files", nargs="+", default=["anchor_dataset_binary_qa.json"]
    )
    parser.add_argument(
        "--answerer_model_path",
        default='qwen3_32b',
        help="Model name for API usage (e.g. meta-llama/Llama-3.2-11B-Vision-Instruct or gpt-4o)",
    )
    parser.add_argument(
        "--answerer_api_base",
        default='http://localhost:8000/v1',
        help="Base URL for answerer API (e.g. http://localhost:8000/v1 or https://api.openai.com/v1)",
    )
    parser.add_argument("--output_file", default="inference_results.json")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=512)

    parser.add_argument(
        "--max_workers",
        type=int,
        default=32,
        help="Maximum number of parallel workers for inference (default: 32)",
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

    inference_engine = SpatialCollabInference(
        answerer_api_base=args.answerer_api_base,
        answerer_model_name=args.answerer_model_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Helper function to load images (runs in thread pool)
    def load_images(example: Dict[str, Any]):
        """Load images synchronously in thread pool."""
        user_1_path = example["user_1_image_local_path"]
        user_2_path = example["user_2_image_local_path"]

        # Handle relative paths (prefixed with data_dir) or absolute paths with replacement
        if user_1_path.startswith("images/"):
            user_1_image_path = os.path.join(args.data_dir, user_1_path)
        else:
            user_1_image_path = user_1_path.replace(
                "/path/to/scratch", "/path/to/scratch"
            )

        if user_2_path.startswith("images/"):
            user_2_image_path = os.path.join(args.data_dir, user_2_path)
        else:
            user_2_image_path = user_2_path.replace(
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
            # Handle case where keys might differ slightly or fallback to user 1
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

    # Helper function to process a single sample
    def process_single_sample(idx: int, example: Dict[str, Any]):
        try:
            # Load images
            loaded_data = load_images(example)

            # Run inference
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
            result["idx"] = idx
            return result
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback

            traceback.print_exc()
            return None

    # Run tasks with progress bar using ThreadPoolExecutor
    results = []
    total_accuracy = 0.0

    print(f"Running inference with {args.max_workers} parallel workers...")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_single_sample, idx, example)
            for idx, example in enumerate(all_data)
        ]

        # Execute with progress bar
        with tqdm(total=len(futures), desc="Running inference") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                    total_accuracy += result["accuracy"]

                pbar.update(1)

                # Print intermediate accuracy every 10 samples
                if len(results) > 0 and len(results) % 10 == 0:
                    current_acc = total_accuracy / len(results)
                    pbar.set_postfix({"accuracy": f"{current_acc:.4f}"})

    # Sort results by original index to maintain order
    results.sort(key=lambda x: x["idx"])
    for result in results:
        del result["idx"]  # Remove temporary index field

    final_accuracy = total_accuracy / len(results) if results else 0.0

    print(f"\n{'='*60}")
    print(f"Inference Complete!")
    print(f"Total Samples: {len(results)}")
    print(f"Overall Accuracy: {final_accuracy:.4f} ({total_accuracy}/{len(results)})")

    print(f"{'='*60}")

    output_data = {
        "config": {
            "answerer_model_path": args.answerer_model_path,
            "num_samples": len(results),
            "max_turns": args.max_turns,
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


if __name__ == "__main__":
    main()
