#!/usr/bin/env python3
"""
Upload MindCube inference results to Weights & Biases.
Creates a table with input images, generated image, question,
predicted/gt answer, and reasoning text.
"""

import argparse
import json
import os

import wandb


def load_results(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def build_id_to_images(dataset_file: str) -> dict:
    """Build a lookup from sample_id -> list of relative image paths."""
    id_to_images = {}
    if not os.path.exists(dataset_file):
        print(f"Warning: dataset file not found: {dataset_file}")
        return id_to_images
    with open(dataset_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            id_to_images[entry["id"]] = entry.get("images", [])
    return id_to_images


def create_wandb_table(
    results_data: dict,
    id_to_images: dict,
    data_dir: str,
    max_samples: int = None,
) -> wandb.Table:
    columns = [
        "sample_id",
        "question",
        "input_images",
        "generated_image",
        "predicted_answer",
        "gt_answer",
        "correct",
        "answer_text",
        "category",
    ]
    table = wandb.Table(columns=columns)

    results = results_data.get("results", [])
    if max_samples is not None:
        results = results[:max_samples]

    for result in results:
        sample_id = result.get("sample_id", "")

        # Input images
        input_images = []
        for rel_path in id_to_images.get(sample_id, []):
            full_path = os.path.join(data_dir, rel_path)
            if os.path.exists(full_path):
                try:
                    input_images.append(wandb.Image(full_path))
                except Exception as e:
                    print(f"Warning: Could not load input image {full_path}: {e}")
            else:
                print(f"Warning: Input image not found: {full_path}")

        # Generated image (visual thinking output)
        gen_image = None
        saved_paths = result.get("saved_image_paths", [])
        if saved_paths and os.path.exists(saved_paths[0]):
            try:
                gen_image = wandb.Image(saved_paths[0])
            except Exception as e:
                print(f"Warning: Could not load generated image {saved_paths[0]}: {e}")

        correct = result.get("accuracy", 0.0) > 0.5
        category = " | ".join(result.get("category", []))

        table.add_data(
            sample_id,
            result.get("question", ""),
            input_images if input_images else None,
            gen_image,
            result.get("predicted_answer", ""),
            result.get("gt_answer", ""),
            correct,
            result.get("final_answer_text", ""),
            category,
        )

    return table


def main():
    parser = argparse.ArgumentParser(
        description="Upload MindCube inference results to Weights & Biases"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="/path/to/scratch/mindcube/BAGEL_format_topdown_qa_visual_only/merged_results.json",
        help="Path to merged_results.json",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/path/to/scratch/datasets/MindCube/data",
        help="Root dir of MindCube dataset (for input images)",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="raw/MindCube_tinybench.jsonl",
        help="Dataset JSONL file relative to data_dir",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="bagel-spatial-reasoning",
        help="W&B project name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="W&B run name (default: parent dir name of results_file)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum number of samples to upload (default: all)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (team or username)",
    )

    args = parser.parse_args()

    print(f"Loading results from {args.results_file}")
    results_data = load_results(args.results_file)

    metrics = results_data.get("metrics", {})
    config = results_data.get("config", {})

    if args.run_name is None:
        args.run_name = os.path.basename(os.path.dirname(args.results_file))

    dataset_path = os.path.join(args.data_dir, args.dataset_file)
    print(f"Pre-loading dataset from {dataset_path}")
    id_to_images = build_id_to_images(dataset_path)

    print(f"Initializing W&B run: {args.run_name}")
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        config={
            "results_file": args.results_file,
            "data_dir": args.data_dir,
            "dataset_file": args.dataset_file,
            **config,
        },
    )

    print(f"Logging metrics: {metrics}")
    wandb.log(metrics)

    print("Creating results table...")
    table = create_wandb_table(
        results_data,
        id_to_images,
        args.data_dir,
        max_samples=args.max_samples,
    )

    print(f"Uploading table with {len(table.data)} rows to W&B...")
    wandb.log({"inference_results": table})

    wandb.finish()
    print(f"Done! View results at: {run.url}")


if __name__ == "__main__":
    main()
