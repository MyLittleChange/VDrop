#!/usr/bin/env python3
"""
Gemini-3-Pro (thinking) oracle ablation inference on the COSMIC eval subset.

Conditions:
  - none           : V1 + V2 only (2-image baseline)
  - T_td_blender   : V1 + V2 + photoreal Cycles ortho top-down (Image 3)
  - T_cor          : V1 + V2 + side-by-side correspondence composite (Image 3)
  - T_pano         : V1 + V2 + 360 panorama (Image 3)

Each oracle condition uses a directive system prompt that explicitly tells
the model to consult Image 3 in its <think> block.

Output format the model is asked to produce:
  <think>step-by-step reasoning ...</think>
  <answer>X</answer>

Reads the eval subset built by build_eval_subset_200.py (one JSON per
subtask, with inlined `_T_td_blender` / `_T_cor_composite` / `_T_pano`
paths per sample).

Outputs:
  <output_dir>/<condition>/<subtask>.json
"""
import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from tqdm import tqdm

CONDITIONS = (
    "none", "T_td_blender", "T_cor", "T_pano",
    # Null/control: random-scene panorama as Image 3, with the T_pano prompt verbatim.
    "T_noise",
    # Generated-image counterparts (BAGEL bridge-masked output as Image 3):
    "T_pano_gen", "T_cor_gen", "T_td_gen",
    # R4 photoreal room-overview (camera at cam0/cam1 XY midpoint raised to
    # ceiling, 10mm wide lens looking at the FOV-overlap region).
    "T_cor_view",
)
SUBTASKS = ("anchor", "counting", "relative_distance", "relative_direction")

DEFAULT_SUBSET_DIR = "/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/eval_subset_200"
DEFAULT_OUTPUT_DIR = "/path/to/scratch/VisualCoT/eval_results/two_reader_oracle_gemini"

ORACLE_FIELDS = {
    "T_td_blender": "_T_td_blender",
    "T_cor": "_T_cor_composite",
    "T_pano": "_T_pano",
    "T_noise": "_T_noise",
    # Model-generated bridges from BAGEL bridge-masked checkpoints:
    "T_pano_gen": "_T_pano_gen",
    "T_cor_gen":  "_T_cor_gen",
    "T_td_gen":   "_T_td_gen",
    # R4 photoreal room-overview corner-view:
    "T_cor_view": "_T_cor_view",
}

SYSTEM_PROMPTS = {
    "none": (
        "You are answering a multiple-choice spatial-reasoning question about an indoor room from two viewpoint images.\n"
        "Think step by step inside <think>...</think>, then output your final answer inside <answer>X</answer> where X is the letter A/B/C/D."
    ),
    "T_td_blender": (
        "You are answering a multiple-choice spatial-reasoning question about an indoor room.\n"
        "- Image 1 and Image 2 show two different viewpoints inside the room.\n"
        "- Image 3 is a photorealistic top-down (bird's-eye) rendering of the same room.\n"
        "Use Image 3 (the top-down) as a map to locate objects in the room and to understand how Image 1's and Image 2's viewpoints relate to each other.\n"
        "IMPORTANT: Your <think> block must explicitly reference what Image 3 shows about the relevant objects' positions. Do not rely only on Images 1 and 2.\n"
        "Think step by step inside <think>...</think>, then output your final answer inside <answer>X</answer> where X is the letter A/B/C/D."
    ),
    "T_cor": (
        "You are answering a multiple-choice spatial-reasoning question about an indoor room.\n"
        "- Image 1 and Image 2 show two different viewpoints inside the room.\n"
        "- Image 3 is a side-by-side composite: the left half is Image 1 and the right half is Image 2, both annotated with coloured dots. Matching colours mark the same real-world object across the two viewpoints (e.g. a red dot on the left half and a red dot on the right half mark the SAME object seen from two angles).\n"
        "Use Image 3's coloured correspondences to align objects across the two viewpoints before reasoning.\n"
        "IMPORTANT: Your <think> block must explicitly reference which colour in Image 3 corresponds to which object.\n"
        "Think step by step inside <think>...</think>, then output your final answer inside <answer>X</answer> where X is the letter A/B/C/D."
    ),
    "T_pano": (
        "You are answering a multiple-choice spatial-reasoning question about an indoor room.\n"
        "- Image 1 and Image 2 show two different viewpoints inside the room.\n"
        "- Image 3 is a 360 panoramic view of the same room, covering the combined field of view of both viewpoints.\n"
        "Use Image 3 (the panorama) to understand the overall room layout and how Image 1's and Image 2's viewpoints relate in 3D.\n"
        "IMPORTANT: Your <think> block must explicitly reference what Image 3 shows about the relative positions of objects mentioned in Image 1 and Image 2.\n"
        "Think step by step inside <think>...</think>, then output your final answer inside <answer>X</answer> where X is the letter A/B/C/D."
    ),
    "T_cor_view": (
        "You are answering a multiple-choice spatial-reasoning question about an indoor room.\n"
        "- Image 1 and Image 2 show two different viewpoints inside the room.\n"
        "- Image 3 is a photorealistic wide-angle room-overview render taken from a camera positioned at the midpoint of the two viewpoints, raised to ceiling height (ceiling cut away), looking down at the region where Image 1's and Image 2's fields of view overlap. The whole room is visible at once like a real-estate listing photo.\n"
        "Use Image 3 (the room overview) to understand the overall room layout, where furniture is, and how Image 1's and Image 2's viewpoints relate in 3D.\n"
        "IMPORTANT: Your <think> block must explicitly reference what Image 3 shows about the relevant objects' positions.\n"
        "Think step by step inside <think>...</think>, then output your final answer inside <answer>X</answer> where X is the letter A/B/C/D."
    ),
}

# Generated-image conditions reuse the GT prompts verbatim — only the
# 3rd image's source differs (GT render vs. BAGEL bridge-masked generation).
SYSTEM_PROMPTS["T_pano_gen"] = SYSTEM_PROMPTS["T_pano"]
SYSTEM_PROMPTS["T_cor_gen"]  = SYSTEM_PROMPTS["T_cor"]
SYSTEM_PROMPTS["T_td_gen"]   = SYSTEM_PROMPTS["T_td_blender"]
# T_noise uses the T_pano prompt verbatim — only the 3rd image's content
# differs (random-scene panorama). Identical prompt isolates "is the model
# using the panorama's content or just any 3rd image?".
SYSTEM_PROMPTS["T_noise"]    = SYSTEM_PROMPTS["T_pano"]


# Answer-extraction regexes
_ANSWER_RE_TAG = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)
_ANSWER_RE_FINAL = re.compile(r"(?:Final\s+Answer|Answer):\s*([A-D])", re.IGNORECASE)
_ANSWER_RE_BRACKET = re.compile(r"[\(\[]([A-D])[\)\]]")
_ANSWER_RE_LETTER = re.compile(r"\b([A-D])\b")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def remap_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    path = path.replace("/path/to/scratch", "/path/to/scratch")
    path = path.replace("/path/to/scratch", "/path/to/scratch")
    return path


def extract_answer(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.replace("**", "").strip()
    m = _ANSWER_RE_TAG.search(t)
    if m:
        return m.group(1).upper()
    m = _ANSWER_RE_FINAL.search(t)
    if m:
        return m.group(1).upper()
    m = _ANSWER_RE_BRACKET.search(t)
    if m:
        return m.group(1).upper()
    letters = _ANSWER_RE_LETTER.findall(t)
    if letters:
        return letters[-1].upper()
    return None


def calculate_accuracy(answer: Optional[str], gold_idx: int) -> float:
    if answer is None or gold_idx is None:
        return 0.0
    return 1.0 if answer == chr(65 + int(gold_idx)) else 0.0


def parse_response_for_thoughts(response) -> tuple[str, str]:
    """Extract (model_response_text, thinking_text).

    When `include_thoughts=True`, Gemini returns its internal thinking as
    `part.thought=True` parts. We concatenate both into output text and
    separate the thinking text into `thinking_text`.
    """
    if response is None:
        return "", ""
    model_text_parts: List[str] = []
    thinking_parts: List[str] = []
    try:
        candidates = response.candidates or []
        if candidates:
            content = candidates[0].content
            for part in content.parts or []:
                if getattr(part, "thought", False):
                    thinking_parts.append(part.text or "")
                else:
                    if part.text:
                        model_text_parts.append(part.text)
    except Exception:
        pass
    model_text = "".join(model_text_parts).strip()
    if not model_text and getattr(response, "text", None):
        model_text = response.text.strip()
    return model_text, "\n".join(thinking_parts).strip()


class GeminiOracleInference:
    def __init__(self, model_name: str, condition: str,
                 temperature: float, max_tokens: int, thinking_level: str):
        api_key = os.environ.get("VisualCoT_GEMINI")
        if not api_key:
            raise ValueError("VisualCoT_GEMINI environment variable is required")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.condition = condition
        self.system_prompt = SYSTEM_PROMPTS[condition]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_level = thinking_level

    def run_one(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sid = sample["sample_id"]

        question = sample.get("user_2_question") or sample.get("user_1_question")
        options = sample.get("options_user_2") or sample.get("options_user_1") or []
        gold_idx = sample.get("user_2_gt_answer_idx")
        if gold_idx is None:
            gold_idx = sample.get("user_1_gt_answer_idx")
        gold_text = sample.get("user_2_gt_answer_text") or sample.get("user_1_gt_answer_text", "")

        if not question or not options or gold_idx is None:
            return {"sample_id": sid, "subtask": sample.get("question_type", ""),
                    "status": "missing_gold"}

        options_str = "\n".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))
        full_question = f"\nQUESTION: {question}\n\nOPTIONS:\n{options_str}"

        # Load V1, V2
        try:
            with open(remap_path(sample["user_1_image_local_path"]), "rb") as f:
                img1 = f.read()
            with open(remap_path(sample["user_2_image_local_path"]), "rb") as f:
                img2 = f.read()
        except FileNotFoundError as e:
            return {"sample_id": sid, "status": "image_missing", "error": str(e)}

        contents: List[Any] = [
            types.Part.from_bytes(data=img1, mime_type="image/png"),
            types.Part.from_bytes(data=img2, mime_type="image/png"),
        ]
        third_path = None
        if self.condition != "none":
            third_path = sample.get(ORACLE_FIELDS[self.condition])
            if not third_path or not Path(third_path).exists():
                return {
                    "sample_id": sid, "subtask": sample.get("question_type", ""),
                    "status": "oracle_missing", "condition": self.condition,
                    "third_path": third_path,
                }
            with open(third_path, "rb") as f:
                img3 = f.read()
            # Composite is JPEG, others are PNG.
            mime = "image/jpeg" if str(third_path).lower().endswith((".jpg", ".jpeg")) else "image/png"
            contents.append(types.Part.from_bytes(data=img3, mime_type=mime))

        full_prompt = self.system_prompt + "\n" + full_question
        contents.append(full_prompt)

        # Retry-on-empty loop: thinking variant occasionally returns
        # response.text=None when reasoning consumes the full budget.
        raw = ""
        thinking_text = ""
        finish_reason = "UNKNOWN"
        attempts = 0
        last_error = None
        for attempt in range(3):
            attempts += 1
            try:
                config = types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=self.thinking_level,
                        include_thoughts=True,
                    ),
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                raw, thinking_text = parse_response_for_thoughts(response)
                if response.candidates:
                    finish_reason = str(response.candidates[0].finish_reason)
                if raw:
                    break
            except Exception as e:
                last_error = str(e)
                time.sleep(2 ** attempt)

        if not raw and last_error:
            return {
                "sample_id": sid, "subtask": sample.get("question_type", ""),
                "status": "api_error", "error": last_error,
                "condition": self.condition, "n_attempts": attempts,
            }

        pred = extract_answer(raw)
        acc = calculate_accuracy(pred, gold_idx)

        return {
            "sample_id": sid,
            "subtask": sample.get("question_type", ""),
            "scene_id": sample.get("scene_id"),
            "condition": self.condition,
            "third_image_path": third_path,
            "predicted_answer": pred,
            "gold_letter": chr(65 + int(gold_idx)),
            "gold_text": gold_text,
            "accuracy": acc,
            "final_answer_text": raw,
            "thinking_text": thinking_text,
            "finish_reason": finish_reason,
            "n_attempts": attempts,
            "status": "ok",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gemini-3-pro-preview")
    parser.add_argument("--thinking_level", default="high", choices=["low", "high"])
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--subtasks", nargs="*", default=list(SUBTASKS))
    parser.add_argument("--subset_dir", default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = GeminiOracleInference(
        model_name=args.model_name,
        condition=args.condition,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking_level=args.thinking_level,
    )

    for subtask in args.subtasks:
        subset_path = Path(args.subset_dir) / f"{subtask}.json"
        if not subset_path.exists():
            print(f"[{subtask}] subset file missing: {subset_path}")
            continue
        samples = json.load(open(subset_path))
        if args.max_samples is not None:
            samples = samples[: args.max_samples]

        out_path = out_dir / f"{subtask}.json"
        done = set()
        prev_results: List[Dict[str, Any]] = []
        if out_path.exists() and not args.no_resume:
            try:
                prev_results = json.load(open(out_path)).get("results", [])
                done = {r["sample_id"] for r in prev_results if r.get("status") == "ok"}
                print(f"[{subtask}] resuming with {len(done)} done")
            except Exception as e:
                print(f"[{subtask}] could not resume: {e}")

        todo = [s for s in samples if s["sample_id"] not in done]
        print(f"[{subtask}] {len(todo)}/{len(samples)} to process")

        all_results: List[Dict[str, Any]] = list(prev_results)
        lock = threading.Lock()

        def _flush():
            with lock:
                payload = {
                    "condition": args.condition,
                    "subtask": subtask,
                    "model_name": args.model_name,
                    "thinking_level": args.thinking_level,
                    "n_ok": sum(1 for r in all_results if r.get("status") == "ok"),
                    "n_skipped": sum(1 for r in all_results if r.get("status") != "ok"),
                    "total_correct": sum(r.get("accuracy", 0.0) for r in all_results if r.get("status") == "ok"),
                    "results": all_results,
                }
                tmp = str(out_path) + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp, out_path)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(engine.run_one, s): s["sample_id"] for s in todo}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{args.condition}/{subtask}"):
                row = fut.result()
                with lock:
                    all_results.append(row)
                if len(all_results) % 10 == 0:
                    _flush()
        _flush()

        n_ok = sum(1 for r in all_results if r.get("status") == "ok")
        n_corr = sum(r.get("accuracy", 0.0) for r in all_results if r.get("status") == "ok")
        n_skip = len(all_results) - n_ok
        acc = (n_corr / n_ok) if n_ok else 0.0
        print(f"[{subtask}] done. ok={n_ok} skipped={n_skip} accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
