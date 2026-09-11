#!/usr/bin/env python3
"""
T_cot annotator for the Two-Reader Informativeness experiment (COSMIC test
split). Prompts Qwen3-VL on (V1, V2, question, options, gold answer) and
emits a `<think>...</think>` block per sample.

Standalone (does NOT reuse the production annotate_text_reasoning.py because
that one requires per-scene metadata JSONs that do not exist for the
v9_dataset_filtered scenes the COSMIC test split lives in).

Output: <output_dir>/<subtask>.jsonl with one row per sample:
    {"sample_id": "...", "subtask": "...", "think_text": "...",
     "predicted_answer": "X", "gold_answer": "Y", "status": "ok"|"wrong"|"parse_fail"}

Quality filter at eval time: keep only `status == "ok"` rows (predicted == gold).
"""
import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

TEST_JSONS = {
    "anchor": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
    "counting": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
    "relative_distance": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
    "relative_direction": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
}

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


def encode_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def build_messages(sample):
    # COSMIC schema: question/options/gold always live in user_2_* fields,
    # regardless of asking_to. The asking_to field only indicates which
    # agent's perspective is the "asker".
    question = sample.get("user_2_question")
    options = sample.get("options_user_2") or []
    gold_idx = sample.get("user_2_gt_answer_idx")
    gold_text = sample.get("user_2_gt_answer_text")
    if gold_idx is None or not options or not question:
        return None
    gold_letter = chr(ord("A") + int(gold_idx))
    options_block = "\n".join(f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options))

    prompt = PROMPT_TEMPLATE.format(
        question=question,
        options_block=options_block,
        gold_letter=gold_letter,
        gold_text=gold_text,
    )

    img1_b64 = encode_image_b64(sample["user_1_image_local_path"])
    img2_b64 = encode_image_b64(sample["user_2_image_local_path"])

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img1_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img2_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "gold_letter": gold_letter,
        "gold_text": gold_text,
    }


def call_with_retry(client, model, messages, temperature, max_tokens, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def parse_response(raw, gold_letter):
    raw = raw or ""
    think_m = THINK_RE.search(raw)
    ans_m = ANSWER_RE.search(raw)
    pred = ans_m.group(1).upper() if ans_m else None

    if think_m:
        think_text = think_m.group(1).strip()
    elif ans_m:
        # Model dropped <think>...</think> tags — fall back to whatever
        # came before <answer>X</answer>.
        before = raw[: ans_m.start()].strip()
        think_text = before if before else None
    else:
        think_text = None

    if think_text and pred == gold_letter:
        status = "ok"
    elif not think_text or not pred:
        status = "parse_fail"
    else:
        status = "wrong"
    return think_text, pred, status


def process_sample(client, model, sample, subtask, temperature, max_tokens):
    sid = sample["sample_id"]
    try:
        prepared = build_messages(sample)
        if prepared is None:
            return {"sample_id": sid, "subtask": subtask, "status": "missing_gold"}
        raw = call_with_retry(client, model, prepared["messages"], temperature, max_tokens)
        think_text, pred, status = parse_response(raw, prepared["gold_letter"])
        return {
            "sample_id": sid,
            "subtask": subtask,
            "think_text": think_text,
            "predicted_answer": pred,
            "gold_answer": prepared["gold_letter"],
            "status": status,
            "raw": raw if status != "ok" else None,
        }
    except Exception as e:
        return {"sample_id": sid, "subtask": subtask, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_base", default="http://localhost:8765/v1")
    parser.add_argument("--model_name", default="qwen3_vl_235b_fp8")
    parser.add_argument(
        "--output_dir",
        default="/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/text_thinking_approved_mcqs",
    )
    parser.add_argument("--subtask", default=None, help="If set, only run that subtask.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--shard", default=None, help="e.g. '0/4'")
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.api_base, api_key="EMPTY")

    subtasks = [args.subtask] if args.subtask else list(TEST_JSONS.keys())

    for subtask in subtasks:
        samples = json.load(open(TEST_JSONS[subtask]))

        # Resume: skip sample_ids already present in the output JSONL
        out_path = out_dir / f"{subtask}.jsonl"
        already_done = set()
        if out_path.exists() and not args.no_resume:
            with open(out_path) as f:
                for ln in f:
                    try:
                        already_done.add(json.loads(ln)["sample_id"])
                    except Exception:
                        pass
            print(f"[{subtask}] resuming with {len(already_done)} done")

        # Shard (deterministic by sample_id sort)
        samples = sorted(samples, key=lambda x: x["sample_id"])
        if args.shard:
            si, st = (int(x) for x in args.shard.split("/"))
            samples = [s for i, s in enumerate(samples) if i % st == si]
        samples = [s for s in samples if s["sample_id"] not in already_done]
        if args.max_samples:
            samples = samples[: args.max_samples]
        if not samples:
            print(f"[{subtask}] nothing to do")
            continue
        print(f"[{subtask}] {len(samples)} to process, workers={args.workers}")

        n_ok = n_wrong = n_fail = n_err = 0
        with open(out_path, "a") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(process_sample, client, args.model_name, s, subtask, args.temperature, args.max_tokens)
                for s in samples
            ]
            for i, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                st = row.get("status")
                if st == "ok":
                    n_ok += 1
                elif st == "wrong":
                    n_wrong += 1
                elif st == "error":
                    n_err += 1
                else:
                    n_fail += 1
                if i % 50 == 0:
                    print(f"  [{subtask}] {i}/{len(samples)} ok={n_ok} wrong={n_wrong} parse_fail={n_fail} err={n_err}")
        print(f"[{subtask}] done: ok={n_ok} wrong={n_wrong} parse_fail={n_fail} err={n_err}")


if __name__ == "__main__":
    main()
