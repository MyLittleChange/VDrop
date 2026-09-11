#!/usr/bin/env python3
"""
Gemini-API fallback annotator for the 2 (or so) T_cot samples that the
Qwen3-VL annotator couldn't get right. Uses gemini-2.5-flash. Appends OK
results to the existing per-subtask JSONL so build_two_reader_index.py
picks them up on the next refresh.
"""
import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*([A-Z])\s*</answer>", re.IGNORECASE)

PROMPT_TEMPLATE = """You are a careful spatial-reasoning assistant. Two camera images of the same indoor scene are provided.

QUESTION: {question}

OPTIONS:
{options_block}

The correct answer is {gold_letter}) {gold_text}.

Walk through the spatial reasoning that leads to {gold_letter} step by step. Reference visible objects in image 1 and image 2 explicitly. Be concrete and concise — 4 to 8 sentences. Do not hedge or list multiple possibilities; commit to the reasoning that supports {gold_letter}.

Output exactly:
<think>your step-by-step reasoning here</think>
<answer>{gold_letter}</answer>
"""

TEST_JSONS = {
    "anchor": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
    "counting": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
    "relative_distance": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
    "relative_direction": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
}

OUT_DIR = Path("/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/text_thinking_approved_mcqs")


def parse_response(raw, gold_letter):
    raw = raw or ""
    think_m = THINK_RE.search(raw)
    ans_m = ANSWER_RE.search(raw)
    pred = ans_m.group(1).upper() if ans_m else None
    if think_m:
        think = think_m.group(1).strip()
    elif ans_m:
        think = raw[: ans_m.start()].strip() or None
    else:
        think = None
    if think and pred == gold_letter:
        status = "ok"
    elif not think or not pred:
        status = "parse_fail"
    else:
        status = "wrong"
    return think, pred, status


def already_ok(out_path, sample_id):
    if not out_path.exists():
        return False
    for ln in open(out_path):
        try:
            r = json.loads(ln)
            if r.get("sample_id") == sample_id and r.get("status") == "ok":
                return True
        except Exception:
            pass
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--max_retries", type=int, default=4)
    args = parser.parse_args()

    api_key = os.environ.get("VisualCoT_GEMINI") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set VisualCoT_GEMINI or GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    for st, jp in TEST_JSONS.items():
        out_path = OUT_DIR / f"{st}.jsonl"
        all_samples = json.load(open(jp))
        # Find missing-or-wrong sample_ids
        ok_sids = set()
        if out_path.exists():
            for ln in open(out_path):
                try:
                    r = json.loads(ln)
                    if r.get("status") == "ok":
                        ok_sids.add(r["sample_id"])
                except Exception:
                    pass
        targets = [s for s in all_samples if s["sample_id"] not in ok_sids]
        if not targets:
            print(f"[{st}] all OK, skipping")
            continue
        print(f"[{st}] {len(targets)} samples to re-annotate via Gemini")

        for s in targets:
            sid = s["sample_id"]
            q = s.get("user_2_question")
            opts = s.get("options_user_2") or []
            gold_idx = s.get("user_2_gt_answer_idx")
            gold_text = s.get("user_2_gt_answer_text")
            if gold_idx is None or not q or not opts:
                print(f"  {sid}: missing gold/question, skip")
                continue
            gold_letter = chr(ord("A") + int(gold_idx))
            options_block = "\n".join(f"{chr(ord('A')+i)}) {o}" for i, o in enumerate(opts))
            prompt = PROMPT_TEMPLATE.format(
                question=q, options_block=options_block,
                gold_letter=gold_letter, gold_text=gold_text,
            )
            img1 = Image.open(s["user_1_image_local_path"])
            img2 = Image.open(s["user_2_image_local_path"])

            think_text = pred = status = None
            raw = ""
            for attempt in range(args.max_retries):
                try:
                    resp = client.models.generate_content(
                        model=args.model,
                        contents=[prompt, img1, img2],
                        config=types.GenerateContentConfig(
                            temperature=0.4 + 0.1 * attempt,
                            max_output_tokens=4096,
                        ),
                    )
                    raw = resp.text
                    think_text, pred, status = parse_response(raw, gold_letter)
                    if status == "ok":
                        break
                except Exception as e:
                    print(f"  {sid} attempt {attempt}: {e}")
                    time.sleep(2 ** attempt)

            row = {
                "sample_id": sid, "subtask": st,
                "think_text": think_text, "predicted_answer": pred,
                "gold_answer": gold_letter, "status": status,
                "raw": raw if status != "ok" else None,
                "annotator": args.model,
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {sid}: status={status}  pred={pred}  gold={gold_letter}")


if __name__ == "__main__":
    main()
