#!/usr/bin/env python3
"""
Gemini-as-autorater for the Two-Reader generated-vs-GT image similarity.

For each (GT, generated) pair in {pano, cor, td}, calls Gemini-3-Flash with
both images and a 1-5 similarity rubric. Outputs per-pairing per-subtask JSONs.

Output dir:
    /path/to/scratch/VisualCoT/eval_results/image_similarity/gemini_judge/<pairing>/<subtask>.json
"""
import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from tqdm import tqdm

PAIRINGS = {
    "pano": ("_T_pano",         "_T_pano_gen"),
    "cor":  ("_T_cor_composite","_T_cor_gen"),
    "td":   ("_T_td_blender",   "_T_td_gen"),
}
SUBTASKS = ("anchor", "counting", "relative_distance", "relative_direction")

DEFAULT_SUBSET_DIR = "/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/eval_subset_200"
DEFAULT_OUTPUT_DIR = "/path/to/scratch/VisualCoT/eval_results/image_similarity/gemini_judge"


JUDGE_PROMPT_TEMPLATE = """You will see two images of the same indoor room.

- Image 1 is a {gt_description} (the ground-truth reference).
- Image 2 is a model-generated rendition attempting to imitate Image 1's spatial content.

Rate how similar Image 2 is to Image 1 on a scale of 1 to 5:
  1 = unrelated scenes (different rooms or no recognisable shared content)
  2 = same kind of room but very different layout / wrong objects
  3 = recognisably the same scene with significant artifacts or missing objects
  4 = close match with minor differences
  5 = near-identical layout and content

Output exactly:
<score>N</score>
<reason>one short sentence explaining your score</reason>
"""

GT_DESCRIPTIONS = {
    "pano": "360-degree panoramic photoreal render of an indoor room",
    "cor":  "side-by-side composite of two viewpoints of the same room with coloured dots marking co-visible objects",
    "td":   "photorealistic top-down (bird's-eye) render of an indoor room",
}

_SCORE_RE = re.compile(r"<score>\s*([1-5])\s*</score>", re.IGNORECASE)
_REASON_RE = re.compile(r"<reason>\s*(.*?)\s*</reason>", re.IGNORECASE | re.DOTALL)
# Fallback: capture everything after "<reason>" even when the closing tag
# is missing (Gemini sometimes truncates).
_REASON_OPEN_RE = re.compile(r"<reason>\s*(.*)", re.IGNORECASE | re.DOTALL)
_LOOSE_SCORE_RE = re.compile(r"\b([1-5])\b")


def parse_response(raw: str) -> Tuple[Optional[int], Optional[str]]:
    if not raw:
        return None, None
    s = _SCORE_RE.search(raw)
    r = _REASON_RE.search(raw)
    score = int(s.group(1)) if s else None
    reason = r.group(1).strip() if r else None
    if reason is None:
        r2 = _REASON_OPEN_RE.search(raw)
        if r2:
            reason = r2.group(1).strip()
    if score is None:
        m = _LOOSE_SCORE_RE.search(raw)
        if m:
            score = int(m.group(1))
    return score, reason


class Judge:
    def __init__(self, model_name: str, temperature: float = 0.3, max_tokens: int = 512):
        api_key = os.environ.get("VisualCoT_GEMINI")
        if not api_key:
            raise ValueError("VisualCoT_GEMINI not set")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def judge_pair(self, pairing: str, gt_path: str, gen_path: str, retries: int = 3) -> Dict:
        with open(gt_path, "rb") as f:
            gt_bytes = f.read()
        with open(gen_path, "rb") as f:
            gen_bytes = f.read()
        gt_mime = "image/jpeg" if gt_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        gen_mime = "image/jpeg" if gen_path.lower().endswith((".jpg", ".jpeg")) else "image/png"

        prompt = JUDGE_PROMPT_TEMPLATE.format(gt_description=GT_DESCRIPTIONS[pairing])
        contents = [
            types.Part.from_bytes(data=gt_bytes, mime_type=gt_mime),
            types.Part.from_bytes(data=gen_bytes, mime_type=gen_mime),
            prompt,
        ]

        raw = ""
        last_error = None
        for attempt in range(retries):
            try:
                config = types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                raw = resp.text or ""
                raw = raw.strip()
                if raw:
                    break
            except Exception as e:
                last_error = str(e)
                time.sleep(2 ** attempt)
        if not raw and last_error:
            return {"score": None, "reason": None, "raw": "", "status": "api_error", "error": last_error, "n_attempts": retries}
        score, reason = parse_response(raw)
        status = "ok" if score is not None else "parse_fail"
        return {"score": score, "reason": reason, "raw": raw, "status": status, "n_attempts": attempt + 1}


def collect_pairs(subset_path: str, pairing: str) -> List[Dict]:
    gt_field, gen_field = PAIRINGS[pairing]
    samples = json.load(open(subset_path))
    pairs = []
    for s in samples:
        gt = s.get(gt_field); gen = s.get(gen_field)
        if not gt or not gen or not Path(gt).exists() or not Path(gen).exists():
            continue
        pairs.append({
            "sample_id": s["sample_id"],
            "subtask": s.get("question_type", ""),
            "scene_id": s.get("scene_id"),
            "pairing": pairing,
            "gt_image_path": gt,
            "generated_image_path": gen,
        })
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairing", default="all", choices=list(PAIRINGS.keys()) + ["all"])
    parser.add_argument("--subtasks", nargs="*", default=list(SUBTASKS))
    parser.add_argument("--subset_dir", default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_name", default="gemini-3-flash-preview")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    pairings_to_run = list(PAIRINGS.keys()) if args.pairing == "all" else [args.pairing]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    judge = Judge(args.model_name, args.temperature, args.max_tokens)
    print(f"[judge] model={args.model_name}  workers={args.workers}")

    for pairing in pairings_to_run:
        for subtask in args.subtasks:
            subset_path = Path(args.subset_dir) / f"{subtask}.json"
            if not subset_path.exists():
                continue
            pairs = collect_pairs(str(subset_path), pairing)
            if args.num_samples is not None:
                pairs = pairs[: args.num_samples]
            if not pairs:
                continue

            out_dir = output_root / pairing
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{subtask}.json"

            done_ids = set()
            prev_results: List[Dict] = []
            if out_path.exists() and not args.no_resume:
                try:
                    prev = json.load(open(out_path))
                    prev_results = prev.get("results", [])
                    done_ids = {r["sample_id"] for r in prev_results if r.get("status") == "ok"}
                except Exception:
                    pass
            todo = [p for p in pairs if p["sample_id"] not in done_ids]
            print(f"[{pairing}/{subtask}] total={len(pairs)} resume={len(done_ids)} todo={len(todo)}")
            if not todo:
                continue

            results: List[Dict] = list(prev_results)
            lock = threading.Lock()

            def _flush():
                payload = {
                    "config": {
                        "pairing": pairing, "subtask": subtask,
                        "model_name": args.model_name,
                        "temperature": args.temperature,
                    },
                    "n_rows": len(results),
                    "results": results,
                }
                tmp = str(out_path) + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp, out_path)

            def _one(p):
                res = judge.judge_pair(pairing, p["gt_image_path"], p["generated_image_path"])
                return {
                    "sample_id": p["sample_id"],
                    "subtask": p["subtask"],
                    "scene_id": p["scene_id"],
                    "pairing": pairing,
                    "gt_image_path": p["gt_image_path"],
                    "generated_image_path": p["generated_image_path"],
                    **res,
                }

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_one, p) for p in todo]
                for i, fut in enumerate(tqdm(as_completed(futs), total=len(futs), desc=f"{pairing}/{subtask}")):
                    row = fut.result()
                    with lock:
                        results.append(row)
                    if (i + 1) % 10 == 0:
                        with lock:
                            _flush()
            with lock:
                _flush()

            n_ok = sum(1 for r in results if r.get("status") == "ok")
            scores = [r["score"] for r in results if r.get("status") == "ok" and r.get("score") is not None]
            mean = (sum(scores) / len(scores)) if scores else 0.0
            print(f"[{pairing}/{subtask}] ok={n_ok}/{len(results)}  mean_score={mean:.2f}")


if __name__ == "__main__":
    main()
