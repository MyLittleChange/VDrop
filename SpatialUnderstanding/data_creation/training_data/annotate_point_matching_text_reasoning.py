#!/usr/bin/env python3
"""
Annotate point-matching `no_thinking.jsonl` with text-only reasoning traces.

Source rows live at e.g.
    /path/to/scratch/infinigen/training_data_point_matching/no_thinking/no_thinking.jsonl

Each row has id ``pointmatch_{v00|v4|v5}_{scene_id}_{qidx}`` and image paths
under ``q/USERNAME/VisualCoT/novel_qa_point_matching{,_v4,_v5}/{room_part}/
{scene_id}/novel_qa/point_matching_img{1,2}_marked_{qidx}.png``. Per-question
oracle metadata is loaded from ``point_matching_qa_questions.json`` next to
the marked PNGs; per-scene ``visible_objects.json`` (sibling of the scene's
``frames/`` directory) supplies the ref-object 2D bbox used to translate
marker pixel coordinates into qualitative placements on the object's
silhouette.

We call Qwen3-VL via vLLM and write
    <output_dir>/point_matching.jsonl  (and ..._failed.jsonl)
with the gpt turn replaced by ``<think>...</think>\\n<answer>X</answer>``.

This script reuses the network/parse/normalize machinery from
``annotate_text_reasoning.py``.
"""

import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from annotate_text_reasoning import (  # noqa: E402
    CategoryWriter,
    build_empty_repair_prompt,
    build_strict_retry_prompt,
    call_vlm,
    image_to_b64_url,  # noqa: F401  (transitively used by call_vlm)
    is_empty_generation,
    normalize_output,
    shard_rows,
    to_jsonable,
)
from text_reasoning_prompts import point_matching as pm_prompts  # noqa: E402
from text_reasoning_prompts.common import SHARED_SYSTEM  # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

NETWORK_SCRATCH = "/network/scratch"
SCRATCH_ROOT = f"{NETWORK_SCRATCH}/q/USERNAME"
INFINIGEN_ROOT = f"{SCRATCH_ROOT}/infinigen"

DEFAULT_INPUT = f"{INFINIGEN_ROOT}/training_data_point_matching/no_thinking/no_thinking.jsonl"
DEFAULT_OUTPUT_DIR = f"{INFINIGEN_ROOT}/training_data_point_matching/text_thinking"


def absolute_image_path(rel_path: str) -> str:
    if rel_path.startswith("/"):
        return rel_path
    return os.path.join(NETWORK_SCRATCH, rel_path)


# ---------------------------------------------------------------------------
# Per-question metadata + per-scene visible_objects loaders (LRU caches)
# ---------------------------------------------------------------------------

class _LRUJsonCache:
    def __init__(self, max_entries: int = 256):
        self._lock = threading.Lock()
        self._data: Dict[str, dict] = {}
        self._order: List[str] = []
        self._max = max_entries

    def get(self, path: str) -> Optional[dict]:
        with self._lock:
            if path in self._data:
                return self._data[path]
        if not os.path.exists(path):
            value = None
        else:
            try:
                with open(path) as f:
                    value = json.load(f)
            except Exception as e:
                print(f"[WARN] failed to load {path}: {e}")
                value = None
        with self._lock:
            self._data[path] = value
            self._order.append(path)
            while len(self._order) > self._max:
                old = self._order.pop(0)
                self._data.pop(old, None)
        return value


def parse_pointmatch_id(sample_id: str) -> Tuple[Optional[str], Optional[int]]:
    """``pointmatch_{src}_{scene_id}_{qidx}`` → (src, qidx)."""
    if not sample_id.startswith("pointmatch_"):
        return None, None
    parts = sample_id.split("_")
    if len(parts) < 4:
        return None, None
    src = parts[1]
    try:
        qidx = int(parts[-1])
    except ValueError:
        return src, None
    return src, qidx


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------

class PointMatchingAnnotator:
    def __init__(self, args, qa_cache: _LRUJsonCache, vis_cache: _LRUJsonCache,
                 client: OpenAI, model_name: str):
        self.args = args
        self.qa_cache = qa_cache
        self.vis_cache = vis_cache
        self.client = client
        self.model = model_name

    def annotate_row(self, row: dict) -> dict:
        sample_id = row["id"]
        src, qidx = parse_pointmatch_id(sample_id)
        if src is None or qidx is None:
            return self._fail(row, f"cannot parse id {sample_id}")

        rel_img1 = row["image"][0]
        img1_abs = absolute_image_path(rel_img1)
        img2_abs = absolute_image_path(row["image"][1])
        if not (os.path.exists(img1_abs) and os.path.exists(img2_abs)):
            return self._fail(row, f"image missing: {img1_abs} / {img2_abs}")

        novel_qa_dir = os.path.dirname(img1_abs)
        qa_json_path = os.path.join(novel_qa_dir, "point_matching_qa_questions.json")
        qa_data = self.qa_cache.get(qa_json_path)
        if qa_data is None:
            return self._fail(row, f"missing QA json: {qa_json_path}")

        questions = qa_data.get("point_matching_questions") or []
        if qidx >= len(questions):
            return self._fail(row, f"qidx {qidx} out of range ({len(questions)})")
        q = questions[qidx]

        gt_letter = (q.get("correct_answer") or "").strip().upper()
        if gt_letter not in {"A", "B", "C", "D"}:
            ci = q.get("correct_index")
            if isinstance(ci, int) and 0 <= ci < 4:
                gt_letter = chr(65 + ci)
            else:
                return self._fail(row, f"no GT letter on question {qidx}")

        options = q.get("options") or ["A", "B", "C", "D"]
        question_text = q.get("question") or row["conversations"][0]["value"]

        hw = qa_data.get("HW") or [720, 1280]

        # visible_objects.json is two levels above novel_qa/.
        scene_dir = qa_data.get("scene_dir")
        visible_objects = {}
        if scene_dir:
            vis_path = os.path.join(scene_dir, "visible_objects.json")
            vis_data = self.vis_cache.get(vis_path)
            if vis_data is not None:
                visible_objects = vis_data

        user_text = pm_prompts.build_prompt(
            q, hw, visible_objects, question_text, options, gt_letter,
        )

        out_text, first_debug = self._call_with_retry(
            SHARED_SYSTEM, user_text, [img1_abs, img2_abs],
        )
        first_norm = normalize_output(out_text, gt_letter)
        final_text = out_text
        final_norm = first_norm
        retry_debug = None
        retry_norm = None
        retry_text = ""
        empty_repair_debug = None
        empty_repair_norm = None

        if first_norm["normalization_status"] != "ok":
            if is_empty_generation(out_text, first_norm):
                retry_prompt = build_empty_repair_prompt(user_text, gt_letter)
                retry_temperature = self.args.empty_retry_temperature
                retry_kind = "empty_repair"
            else:
                retry_prompt = build_strict_retry_prompt(user_text, gt_letter)
                retry_temperature = self.args.temperature
                retry_kind = "strict_retry"

            retry_text, retry_debug = self._call_with_retry(
                SHARED_SYSTEM, retry_prompt, [img1_abs, img2_abs],
                temperature=retry_temperature,
            )
            retry_debug["retry_kind"] = retry_kind
            retry_debug["request_temperature"] = retry_temperature
            retry_norm = normalize_output(retry_text, gt_letter)
            if retry_norm["normalization_status"] == "ok":
                final_text = retry_text
                final_norm = retry_norm

        if (
            final_norm["normalization_status"] != "ok"
            and retry_norm is not None
            and is_empty_generation(retry_text, retry_norm)
            and not is_empty_generation(out_text, first_norm)
        ):
            repair_prompt = build_empty_repair_prompt(user_text, gt_letter)
            repair_text, empty_repair_debug = self._call_with_retry(
                SHARED_SYSTEM, repair_prompt, [img1_abs, img2_abs],
                temperature=self.args.empty_retry_temperature,
            )
            empty_repair_debug["retry_kind"] = "empty_repair_after_strict_retry"
            empty_repair_debug["request_temperature"] = self.args.empty_retry_temperature
            empty_repair_norm = normalize_output(repair_text, gt_letter)
            if empty_repair_norm["normalization_status"] == "ok":
                final_text = repair_text
                final_norm = empty_repair_norm

        if final_norm["normalization_status"] != "ok":
            return self._fail(
                row,
                (
                    "final mismatch: "
                    f"parsed answer={final_norm.get('answer_letter')}, gt={gt_letter}, "
                    f"status={final_norm.get('normalization_status')}"
                ),
                raw=final_text,
                response_debug={
                    "first_attempt": first_debug,
                    "retry_attempt": retry_debug,
                    "empty_repair_attempt": empty_repair_debug,
                    "first_normalization": first_norm,
                    "retry_normalization": retry_norm,
                    "empty_repair_normalization": empty_repair_norm,
                    "final_normalization": final_norm,
                    "final_content_length": len(final_text or ""),
                    "final_has_think": bool(final_norm.get("think_text")),
                    "final_parsed_answer": final_norm.get("answer_letter"),
                    "gt_answer": gt_letter,
                },
            )

        new_row = dict(row)
        new_row["conversations"] = [
            row["conversations"][0],
            {"from": "gpt", "value": final_norm["normalized_output"]},
        ]
        new_row["_category"] = "point_matching"
        new_row["_status"] = "ok"
        return new_row

    def _call_with_retry(self, system_text: str, user_text: str,
                         image_paths: List[str],
                         temperature: Optional[float] = None) -> Tuple[str, dict]:
        last_err = None
        request_temperature = (
            self.args.temperature if temperature is None else temperature
        )
        for attempt in range(self.args.network_retries):
            try:
                text, debug = call_vlm(
                    self.client, self.model, system_text, user_text, image_paths,
                    temperature=request_temperature, max_tokens=self.args.max_tokens,
                )
                debug["network_attempt"] = attempt + 1
                debug["request_temperature"] = request_temperature
                return text, debug
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"vLLM call failed after retries: {last_err}")

    def _fail(self, row: dict, reason: str, raw: str = "",
              response_debug: Optional[dict] = None) -> dict:
        failed = {
            "id": row["id"],
            "image": row["image"],
            "conversations": row["conversations"],
            "_category": "point_matching",
            "_status": "failed",
            "_reason": reason,
            "_raw_output": raw,
        }
        if response_debug is not None:
            failed["_response_debug"] = to_jsonable(response_debug)
        return failed


# ---------------------------------------------------------------------------
# Dump-prompt smoke test (no LLM)
# ---------------------------------------------------------------------------

def dump_prompt(args, qa_cache: _LRUJsonCache, vis_cache: _LRUJsonCache):
    target_id = args.dump_prompt_for_id
    with open(args.input) as f:
        for ln in f:
            row = json.loads(ln)
            if target_id and row["id"] != target_id:
                continue
            sample_id = row["id"]
            src, qidx = parse_pointmatch_id(sample_id)
            if src is None or qidx is None:
                continue
            img1_abs = absolute_image_path(row["image"][0])
            qa_path = os.path.join(os.path.dirname(img1_abs),
                                   "point_matching_qa_questions.json")
            qa_data = qa_cache.get(qa_path)
            if qa_data is None or qidx >= len(qa_data.get("point_matching_questions", [])):
                continue
            q = qa_data["point_matching_questions"][qidx]
            hw = qa_data.get("HW") or [720, 1280]
            visible_objects = {}
            scene_dir = qa_data.get("scene_dir")
            if scene_dir:
                vis = vis_cache.get(os.path.join(scene_dir, "visible_objects.json"))
                if vis is not None:
                    visible_objects = vis

            gt_letter = (q.get("correct_answer") or "").strip().upper()
            options = q.get("options") or ["A", "B", "C", "D"]
            question_text = q.get("question") or row["conversations"][0]["value"]
            user_text = pm_prompts.build_prompt(
                q, hw, visible_objects, question_text, options, gt_letter,
            )

            print("=" * 80)
            print(f"id: {sample_id}  src: {src}  qidx: {qidx}")
            print(f"GT: {gt_letter}")
            print("=" * 80)
            print("SYSTEM:")
            print(SHARED_SYSTEM)
            print()
            print("USER:")
            print(user_text)
            return
    print(f"No row matched dump_prompt_for_id={target_id!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api_base", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", default=None,
                        help="If unset, auto-detect from vLLM /v1/models")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--empty_retry_temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=768)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--network_retries", type=int, default=4)
    parser.add_argument("--shard", default=None, help="e.g. '0/8'")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap rows for smoke testing")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--dump_prompt_for_id", default=None,
                        help="If set, print the prompt for that exact sample id "
                        "and exit (no LLM call). Falls back to first row if id "
                        "is empty.")
    args = parser.parse_args()

    qa_cache = _LRUJsonCache(max_entries=512)
    vis_cache = _LRUJsonCache(max_entries=512)

    if args.dump_prompt_for_id is not None:
        dump_prompt(args, qa_cache, vis_cache)
        return

    client = OpenAI(base_url=args.api_base, api_key="EMPTY")
    model_name = args.model_name
    if model_name is None:
        model_name = client.models.list().data[0].id
        print(f"Auto-detected model: {model_name}")

    print(f"Reading {args.input}")
    rows: List[dict] = []
    with open(args.input) as f:
        for ln in f:
            rows.append(json.loads(ln))
    print(f"Total rows: {len(rows)}")

    rows, shard_idx, total_shards = shard_rows(rows, args.shard)
    print(f"Shard {shard_idx}/{total_shards}: {len(rows)} rows")

    if args.max_samples is not None and args.max_samples < len(rows):
        rows = rows[: args.max_samples]
        print(f"Capped to first {len(rows)} rows for smoke test")

    writer = CategoryWriter(args.output_dir)
    if not args.no_resume:
        done = writer.collect_completed_ids()
        if done:
            already = done.get("point_matching", set())
            if already:
                print(f"Resuming: {len(already)} rows already in output")
                rows = [r for r in rows if r["id"] not in already]
                print(f"After resume filter: {len(rows)} rows to process")

    annotator = PointMatchingAnnotator(args, qa_cache, vis_cache, client, model_name)

    ok_n, fail_n = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(annotator.annotate_row, r): r for r in rows}
        with tqdm(total=len(futures), desc="annotate") as pbar:
            for fut in as_completed(futures):
                row = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"[ERROR] worker exception: {e}")
                    res = annotator._fail(
                        row,
                        f"worker exception: {type(e).__name__}: {e}",
                        response_debug={
                            "exception_type": type(e).__name__,
                            "exception_repr": repr(e),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    writer.write(res)
                    fail_n += 1
                    pbar.update(1)
                    continue
                writer.write(res)
                if res["_status"] == "ok":
                    ok_n += 1
                else:
                    fail_n += 1
                pbar.set_postfix({"ok": ok_n, "fail": fail_n})
                pbar.update(1)

    print(f"\nDone. ok={ok_n} fail={fail_n}")
    print(f"Outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
