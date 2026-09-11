#!/usr/bin/env python3
"""
Evaluate text-only thinking outputs against ground truth thinking annotations.

Computes ROUGE-L, BLEU-{1,2,3,4}, and CIDEr metrics.

Requires:
    pip install nltk rouge-score pycocoevalcap

Usage:
    python SpatialUnderstanding/eval_thinking_text_metrics.py \
        --prediction_files \
            /path/to/results_summary_shard0.json \
            /path/to/results_summary_shard1.json \
        --gt_file /path/to/parsed_qa_test.json
"""

import argparse
import json
import re
import os
from collections import defaultdict

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider


def extract_think_text(text: str) -> str:
    """Extract text between <think>...</think> tags. Falls back to full text."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate thinking text with ROUGE/BLEU/CIDEr")
    parser.add_argument(
        "--prediction_files",
        nargs="+",
        default=[
            "/path/to/scratch/VisualCoT/orbit_orbit_annotations_parsed_qa_train_interleaved_thinking_parsed_qa_test/results_summary_shard0.json",
            "/path/to/scratch/VisualCoT/orbit_orbit_annotations_parsed_qa_train_interleaved_thinking_parsed_qa_test/results_summary_shard1.json",
        ],
        help="Path(s) to prediction result summary JSON files (shards)",
    )
    parser.add_argument(
        "--gt_file",
        default="/path/to/scratch/VisualCoT/annotations/gemini_3_flash_preview_rendered_orbit_orbit_annotations_parsed_qa_test.json",
        help="Path to ground truth parsed QA JSON",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Optional path to save metrics JSON (defaults to prediction dir)",
    )
    args = parser.parse_args()

    # Load ground truth
    print(f"Loading ground truth from {args.gt_file}")
    with open(args.gt_file) as f:
        gt_data = json.load(f)
    gt_map = {}
    for item in gt_data["results"]:
        sid = item["sample_id"]
        gt_map[sid] = item.get("thinking_annotation", "").strip()
    print(f"  Ground truth samples: {len(gt_map)}")

    # Load predictions from all shard files
    predictions = {}
    for pred_file in args.prediction_files:
        print(f"Loading predictions from {pred_file}")
        with open(pred_file) as f:
            pred_data = json.load(f)
        for item in pred_data["results"]:
            sid = item["sample_id"]
            text_outputs = item.get("text_outputs", [])
            # Concatenate all text outputs and extract from <think> tags
            full_text = " ".join(text_outputs)
            predictions[sid] = extract_think_text(full_text)
    print(f"  Total prediction samples: {len(predictions)}")

    # Match predictions to ground truth
    matched_ids = sorted(set(predictions.keys()) & set(gt_map.keys()))
    print(f"  Matched samples: {len(matched_ids)}")

    if not matched_ids:
        print("ERROR: No matching sample_ids between predictions and ground truth.")
        return

    # Prepare references and hypotheses
    references = [gt_map[sid] for sid in matched_ids]
    hypotheses = [predictions[sid] for sid in matched_ids]

    # --- ROUGE-L ---
    print("\nComputing ROUGE-L...")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = []
    for ref, hyp in zip(references, hypotheses):
        score = scorer.score(ref, hyp)
        rouge_scores.append(score["rougeL"].fmeasure)
    avg_rouge_l = sum(rouge_scores) / len(rouge_scores)

    # --- BLEU ---
    print("Computing BLEU...")
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    smoother = SmoothingFunction().method1

    bleu_scores = {1: [], 2: [], 3: [], 4: []}
    weights_map = {
        1: (1.0, 0, 0, 0),
        2: (0.5, 0.5, 0, 0),
        3: (1/3, 1/3, 1/3, 0),
        4: (0.25, 0.25, 0.25, 0.25),
    }

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = nltk.word_tokenize(ref.lower())
        hyp_tokens = nltk.word_tokenize(hyp.lower())
        if not hyp_tokens:
            for n in range(1, 5):
                bleu_scores[n].append(0.0)
            continue
        for n in range(1, 5):
            score = sentence_bleu(
                [ref_tokens], hyp_tokens,
                weights=weights_map[n],
                smoothing_function=smoother,
            )
            bleu_scores[n].append(score)

    avg_bleu = {n: sum(scores) / len(scores) for n, scores in bleu_scores.items()}

    # --- CIDEr ---
    print("Computing CIDEr...")
    # pycocoevalcap expects {id: [str]} format
    gts = {sid: [gt_map[sid]] for sid in matched_ids}
    res = {sid: [predictions[sid]] for sid in matched_ids}
    cider_scorer = Cider()
    cider_score, per_sample_cider = cider_scorer.compute_score(gts, res)

    # --- Print Results ---
    print(f"\n{'='*60}")
    print(f"Text Thinking Evaluation Results")
    print(f"{'='*60}")
    print(f"Samples evaluated: {len(matched_ids)}")
    print(f"")
    print(f"ROUGE-L:  {avg_rouge_l:.4f}")
    print(f"BLEU-1:   {avg_bleu[1]:.4f}")
    print(f"BLEU-2:   {avg_bleu[2]:.4f}")
    print(f"BLEU-3:   {avg_bleu[3]:.4f}")
    print(f"BLEU-4:   {avg_bleu[4]:.4f}")
    print(f"CIDEr:    {cider_score:.4f}")
    print(f"{'='*60}")

    # --- Save results ---
    metrics = {
        "num_samples": len(matched_ids),
        "rouge_l": avg_rouge_l,
        "bleu_1": avg_bleu[1],
        "bleu_2": avg_bleu[2],
        "bleu_3": avg_bleu[3],
        "bleu_4": avg_bleu[4],
        "cider": cider_score,
    }

    # Per-sample results
    per_sample = []
    for i, sid in enumerate(matched_ids):
        per_sample.append({
            "sample_id": sid,
            "rouge_l": rouge_scores[i],
            "bleu_1": bleu_scores[1][i],
            "bleu_4": bleu_scores[4][i],
            "cider": float(per_sample_cider[i]),
            "prediction": predictions[sid],
            "reference": gt_map[sid],
        })

    output = {
        "config": {
            "prediction_files": args.prediction_files,
            "gt_file": args.gt_file,
        },
        "metrics": metrics,
        "per_sample": per_sample,
    }

    if args.output_file:
        output_path = args.output_file
    else:
        output_dir = os.path.dirname(args.prediction_files[0])
        output_path = os.path.join(output_dir, "text_metrics.json")

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
