#!/usr/bin/env python
"""
Re-extract answers from Gemini VisWorld results.

1. Try regex extraction (boxed, bbox, <answer>) first.
2. For remaining None cases, call Gemini to extract the answer from the response.
3. Compare with GroundTruth and report per-category accuracy.
4. Save enriched results to a new JSON file.

Usage:
    python extract_and_eval_gemini.py \
        --input /path/to/results_gemini-3-flash-preview.json \
        --output /path/to/results_gemini-3-flash-preview_evaluated.json
"""

import os
import re
import json
import argparse
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Regex-based extraction
# ---------------------------------------------------------------------------

def extract_boxed_answer(text):
    """Extract the content from the last \\boxed{}, \\bbox{}, or <answer> pattern."""
    if not text:
        return None
    # \boxed{...}
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    # \bbox{...}
    pattern_bbox = r"\\bbox\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches_bbox = re.findall(pattern_bbox, text)
    if matches_bbox:
        return matches_bbox[-1].strip()
    # <answer>...</answer>
    pattern_alt = r"<answer>(.*?)</answer>"
    matches_alt = re.findall(pattern_alt, text, re.DOTALL)
    if matches_alt:
        return matches_alt[-1].strip()
    return None


# ---------------------------------------------------------------------------
# Gemini-based extraction
# ---------------------------------------------------------------------------

_thread_local = threading.local()

def get_client():
    if not hasattr(_thread_local, "client"):
        _thread_local.client = genai.Client()
    return _thread_local.client


EXTRACT_PROMPT_TEMPLATE = """You are an answer extraction assistant. Given a model's response to a question, extract ONLY the final answer. Output just the answer value with no extra text.

Rules:
- If the response states a letter choice (A, B, C, D, etc.), output just the letter.
- If the response states a number, output just the number.
- If the response gives coordinates or a path, output them exactly as given.
- If the response gives a list of moves/directions, output them exactly.
- If no clear answer is found, output: NONE

Question:
{question}

Model's response:
{response}

Ground truth format example: {ground_truth}

Extract the final answer (just the value, nothing else):"""


def gemini_extract_answer(entry, model_name="gemini-3-flash-preview"):
    """Call Gemini to extract the answer from the model response."""
    client = get_client()
    response_text = entry.get("ModelResponse", "")
    # Also include code results if any
    code_results = entry.get("CodeResults", [])
    if code_results:
        response_text = response_text + "\n\nCode execution output:\n" + "\n".join(code_results)

    prompt = EXTRACT_PROMPT_TEMPLATE.format(
        question=entry.get("Question", "")[:500],
        response=response_text[-2000:] if len(response_text) > 2000 else response_text,
        ground_truth=entry.get("GroundTruth", ""),
    )

    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=256,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )
        answer = resp.text.strip() if resp.text else None
        if answer == "NONE":
            return None
        return answer
    except Exception as e:
        print(f"  Gemini extraction error for {entry.get('Id', '?')}: {e}")
        return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def normalize_answer(ans):
    """Normalize an answer string for comparison."""
    if ans is None:
        return None
    ans = str(ans).strip().lower()
    # Remove trailing period
    ans = ans.rstrip(".")
    # Remove surrounding quotes
    if len(ans) >= 2 and ans[0] == ans[-1] and ans[0] in ('"', "'"):
        ans = ans[1:-1]
    return ans


def check_match(extracted, ground_truth):
    """Check if extracted answer matches ground truth."""
    e = normalize_answer(extracted)
    g = normalize_answer(ground_truth)
    if e is None or g is None:
        return False
    return e == g


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Re-extract and evaluate Gemini VisWorld results")
    parser.add_argument(
        "--input",
        type=str,
        default="/path/to/scratch/VisualCoT/visworld_results/Gemini/results_gemini-3-flash-preview.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: input with _evaluated suffix)",
    )
    parser.add_argument(
        "--extraction_model",
        type=str,
        default="gemini-3-flash-preview",
        help="Gemini model for answer extraction",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of parallel workers for Gemini calls",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file for Gemini extractions (default: auto)",
    )
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_evaluated{ext}"

    if args.checkpoint is None:
        args.checkpoint = args.output + ".extraction_ckpt.jsonl"

    # Load data
    print(f"Loading: {args.input}")
    with open(args.input) as f:
        data = json.load(f)
    print(f"Total entries: {len(data)}")

    # ---------------------------------------------------------------
    # Step 1: Regex extraction pass
    # ---------------------------------------------------------------
    print("\n=== Step 1: Regex extraction ===")
    regex_extracted = 0
    need_gemini = []

    for i, entry in enumerate(data):
        response = entry.get("ModelResponse", "")
        code_results = entry.get("CodeResults", [])
        full_text = response + "\n" + "\n".join(code_results) if code_results else response

        ans = extract_boxed_answer(full_text)
        if ans is not None:
            entry["GeminiExtractedAnswer"] = ans
            regex_extracted += 1
        elif not response:
            # Empty response, nothing to extract
            entry["GeminiExtractedAnswer"] = None
        else:
            need_gemini.append(i)

    print(f"  Regex extracted: {regex_extracted}")
    print(f"  Empty responses (skipped): {sum(1 for e in data if not e.get('ModelResponse'))}")
    print(f"  Need Gemini extraction: {len(need_gemini)}")

    # ---------------------------------------------------------------
    # Step 2: Load checkpoint for Gemini extractions
    # ---------------------------------------------------------------
    cached_extractions = {}
    if os.path.exists(args.checkpoint):
        with open(args.checkpoint) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    cached_extractions[rec["Id"]] = rec.get("GeminiExtractedAnswer")
        print(f"  Loaded {len(cached_extractions)} cached Gemini extractions")

    # Apply cached extractions
    still_need = []
    for idx in need_gemini:
        entry_id = data[idx]["Id"]
        if entry_id in cached_extractions:
            data[idx]["GeminiExtractedAnswer"] = cached_extractions[entry_id]
        else:
            still_need.append(idx)

    print(f"  Still need Gemini extraction: {len(still_need)}")

    # ---------------------------------------------------------------
    # Step 3: Gemini extraction for remaining
    # ---------------------------------------------------------------
    if still_need:
        print(f"\n=== Step 2: Gemini extraction ({len(still_need)} entries) ===")
        ckpt_lock = threading.Lock()
        ckpt_file = open(args.checkpoint, "a")
        completed = 0

        def process_one(idx):
            entry = data[idx]
            ans = gemini_extract_answer(entry, model_name=args.extraction_model)
            return idx, ans

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_one, idx): idx for idx in still_need}
            for future in as_completed(futures):
                idx, ans = future.result()
                data[idx]["GeminiExtractedAnswer"] = ans
                completed += 1

                # Write checkpoint
                with ckpt_lock:
                    ckpt_file.write(json.dumps({
                        "Id": data[idx]["Id"],
                        "GeminiExtractedAnswer": ans,
                    }) + "\n")
                    ckpt_file.flush()

                if completed % 100 == 0 or completed == len(still_need):
                    print(f"  [{completed}/{len(still_need)}] Last: {data[idx]['Id']} -> {str(ans)[:80]}")

        ckpt_file.close()

    # ---------------------------------------------------------------
    # Step 4: Evaluate
    # ---------------------------------------------------------------
    print("\n=== Step 3: Evaluation ===")

    for entry in data:
        extracted = entry.get("GeminiExtractedAnswer")
        gt = entry.get("GroundTruth")
        entry["ReEvalExactMatch"] = check_match(extracted, gt)

    # ---------------------------------------------------------------
    # Step 5: Per-category accuracy
    # ---------------------------------------------------------------
    split_stats = defaultdict(lambda: {"total": 0, "correct": 0, "extracted": 0, "empty_response": 0})
    overall = {"total": 0, "correct": 0, "extracted": 0}

    for entry in data:
        s = entry["Split"]
        split_stats[s]["total"] += 1
        overall["total"] += 1

        if not entry.get("ModelResponse"):
            split_stats[s]["empty_response"] += 1
            continue

        if entry.get("GeminiExtractedAnswer") is not None:
            split_stats[s]["extracted"] += 1
            overall["extracted"] += 1

        if entry.get("ReEvalExactMatch"):
            split_stats[s]["correct"] += 1
            overall["correct"] += 1

    print(f"\n{'='*70}")
    print(f"{'Category':<18} {'Correct':>8} {'Total':>8} {'Accuracy':>10} {'Extracted':>10}")
    print(f"{'='*70}")
    for s in sorted(split_stats.keys()):
        stats = split_stats[s]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{s:<18} {stats['correct']:>8} {stats['total']:>8} {acc:>9.1f}% {stats['extracted']:>10}")

    print(f"{'-'*70}")
    acc_all = overall["correct"] / overall["total"] * 100 if overall["total"] > 0 else 0
    print(f"{'OVERALL':<18} {overall['correct']:>8} {overall['total']:>8} {acc_all:>9.1f}% {overall['extracted']:>10}")
    print(f"{'='*70}")

    # ---------------------------------------------------------------
    # Step 6: Save
    # ---------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Clean up checkpoint
    if os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
        print(f"Checkpoint removed: {args.checkpoint}")


if __name__ == "__main__":
    main()
