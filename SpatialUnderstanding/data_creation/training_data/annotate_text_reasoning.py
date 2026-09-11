#!/usr/bin/env python3
"""
Annotate `no_thinking.jsonl` rows with text-only reasoning traces.

For each row in
    /path/to/scratch/infinigen/training_data_mix_all_balance/no_thinking/no_thinking.jsonl
we look up the original sample metadata in the V4 / V5 dataset jsons + per-scene
metadata under {room_part}/{scene_id}/, build a category-specific prompt that
includes the GT answer, send the two camera images + the prompt to a Qwen3-VL
vLLM endpoint, and write a row to
    /path/to/scratch/infinigen/training_data_mix_all_balance/text_thinking/<category>.jsonl
with the gpt value replaced by `<think>...reasoning...</think>\\n<answer>X</answer>`.

Run with sharding:
    python annotate_text_reasoning.py --shard 0/8 --workers 16

The vLLM server must be running first (see scripts/vllm_server_qwen_3_235B_FP8.sh).
"""

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

# Allow `from text_reasoning_prompts import ...` when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_reasoning_prompts import DISPATCH  # noqa: E402
from text_reasoning_prompts.common import SHARED_SYSTEM  # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

NETWORK_SCRATCH = "/network/scratch"
SCRATCH_ROOT = f"{NETWORK_SCRATCH}/q/USERNAME"
INFINIGEN_ROOT = f"{SCRATCH_ROOT}/infinigen"
VISUALCOT_ROOT = f"{SCRATCH_ROOT}/VisualCoT/spatial_collab_dataset"
ANKUR_SCENE_ROOT = (
    f"{NETWORK_SCRATCH}/a/USERNAME/infinigen/infinigen_debang/infinigen/v00_filtered"
)

DEFAULT_INPUT = f"{INFINIGEN_ROOT}/training_data_mix_all_balance/no_thinking/no_thinking.jsonl"
DEFAULT_OUTPUT_DIR = f"{INFINIGEN_ROOT}/training_data_mix_all_balance/text_thinking"

# category → {source: metadata_json_path}
DATASET_FILES = {
    "counting": {
        "v5": f"{INFINIGEN_ROOT}/spatial/dataset_counting_questions_filtered_V5_normalized.json",
        "v4": f"{INFINIGEN_ROOT}/outputs_rendered/dataset_counting_questions_filtered_V4.json",
        "ankur": f"{VISUALCOT_ROOT}/counting_dataset_V_Final_2000.json",
    },
    "anchor": {
        "v5": f"{INFINIGEN_ROOT}/spatial/dataset_anchor_questions_filtered_V5_normalized.json",
        "v4": f"{INFINIGEN_ROOT}/dataset_anchor_questions_filtered_V4_normalized.json",
        "ankur": f"{VISUALCOT_ROOT}/anchor_dataset_V_Final_2000.json",
    },
    "spatial": {
        "v5": f"{INFINIGEN_ROOT}/spatial/dataset_spatial_questions_filtered_V5_normalized.json",
        "v4": f"{INFINIGEN_ROOT}/dataset_spatial_questions_filtered_V4_normalized.json",
        "ankur": f"{VISUALCOT_ROOT}/spatial_dataset_V_Final_2000_normalized.json",
    },
    "relative_distance": {
        "v5": f"{INFINIGEN_ROOT}/spatial/dataset_relative_distance_questions_filtered_V5.json",
        "v4": f"{INFINIGEN_ROOT}/outputs_rendered/dataset_relative_distance_questions_filtered_V4.json",
        "ankur": f"{VISUALCOT_ROOT}/relative_dataset_V_Final_2000.json",
    },
    "perspective_taking": {
        "v5": f"{INFINIGEN_ROOT}/spatial/dataset_perspective_taking_questions_filtered_V5_normalized.json",
        "v4": f"{INFINIGEN_ROOT}/dataset_perspective_taking_questions_filtered_V4_normalized.json",
    },
}

# source → scene_root
SCENE_ROOTS = {
    "v5": f"{INFINIGEN_ROOT}/spatial",
    "v4": f"{INFINIGEN_ROOT}/outputs_rendered",
    "ankur": ANKUR_SCENE_ROOT,
}


# ---------------------------------------------------------------------------
# Lookup tables (built once)
# ---------------------------------------------------------------------------

def _safe_load_json(path: str) -> Optional[list]:
    if not os.path.exists(path):
        print(f"[WARN] missing dataset file: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def build_metadata_lookup() -> Dict[str, Dict[str, Dict]]:
    """Returns {category: {source: {sample_id: row}}}."""
    lookup: Dict[str, Dict[str, Dict]] = {}
    for cat, source_paths in DATASET_FILES.items():
        lookup[cat] = {src: {} for src in source_paths}
        for src, path in source_paths.items():
            data = _safe_load_json(path) or []
            for row in data:
                sid = row.get("sample_id")
                if sid:
                    lookup[cat][src][sid] = row
        counts = ", ".join(
            f"{src.upper()}={len(lookup[cat][src])}"
            for src in sorted(lookup[cat])
        )
        print(f"  {cat}: {counts}")
    return lookup


SOURCE_PREFIXES = ("v4_", "v5_", "ankur_")


def strip_source_prefix(sample_id: str) -> str:
    """Strip a leading 'v4_'/'v5_'/'ankur_' from a sample id, if present.

    Source-prefixed ids are introduced by reprefix_text_thinking_ids.py to
    disambiguate sample_ids that collide across V4/V5/ankur."""
    for p in SOURCE_PREFIXES:
        if sample_id.startswith(p):
            return sample_id[len(p):]
    return sample_id


def category_for_id(sample_id: str) -> str:
    """Map e.g. 'spatial_000123' or 'v4_spatial_000123' → 'spatial'."""
    return strip_source_prefix(sample_id).rsplit("_", 1)[0]


def parse_source_from_image_path(
    image_path: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Parse 'q/USERNAME/infinigen/spatial/<room_part>/<scene_id>/...' or
          'q/USERNAME/infinigen/outputs_rendered/<room_part>/<scene_id>/...' or
          'a/USERNAME/spatial_collab_dataset/scenes/<version>/<room_part>/<scene_id>/...'
    Returns (source, version_folder, room_part, scene_id).
    """
    if "infinigen/spatial/" in image_path:
        rest = image_path.split("infinigen/spatial/", 1)[1]
        parts = rest.split("/")
        if len(parts) >= 2:
            return ("v5", None, parts[0], parts[1])
    if "outputs_rendered/" in image_path:
        rest = image_path.split("outputs_rendered/", 1)[1]
        parts = rest.split("/")
        if len(parts) >= 2:
            return ("v4", None, parts[0], parts[1])
    anchor = "spatial_collab_dataset/scenes/"
    if anchor in image_path:
        rest = image_path.split(anchor, 1)[1]
        parts = rest.split("/")
        if len(parts) >= 3:
            return ("ankur", parts[0], parts[1], parts[2])
    return (None, None, None, None)


def absolute_image_path(rel_path: str) -> str:
    """The jsonl stores paths relative to /network/scratch/."""
    if rel_path.startswith("/"):
        return rel_path
    return os.path.join(NETWORK_SCRATCH, rel_path)


# ---------------------------------------------------------------------------
# Per-scene metadata loading (cached per source / scene location)
# ---------------------------------------------------------------------------

class SceneCache:
    def __init__(self, max_entries: int = 256):
        self._lock = threading.Lock()
        self._data: Dict[Tuple[str, Optional[str], str, str], dict] = {}
        self._order: List[Tuple[str, Optional[str], str, str]] = []
        self._max = max_entries

    def get(self, source: str, version_folder: Optional[str], room_part: str, scene_id: str) -> dict:
        key = (source, version_folder, room_part, scene_id)
        with self._lock:
            if key in self._data:
                return self._data[key]
        if source == "ankur":
            if not version_folder:
                return {}
            scene_dir = os.path.join(SCENE_ROOTS[source], version_folder, room_part, scene_id)
        else:
            scene_dir = os.path.join(SCENE_ROOTS[source], room_part, scene_id)
        scene = {}
        vis_path = os.path.join(scene_dir, "visible_objects_with_descriptions.json")
        if os.path.exists(vis_path):
            with open(vis_path) as f:
                scene["visible"] = json.load(f)
        cam_path = os.path.join(scene_dir, "cameras.json")
        if os.path.exists(cam_path):
            with open(cam_path) as f:
                scene["cameras"] = json.load(f)
        with self._lock:
            self._data[key] = scene
            self._order.append(key)
            while len(self._order) > self._max:
                old = self._order.pop(0)
                self._data.pop(old, None)
        return scene


# ---------------------------------------------------------------------------
# vLLM client
# ---------------------------------------------------------------------------

def image_to_b64_url(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")


def to_jsonable(obj: Any) -> Any:
    """Best-effort conversion for OpenAI SDK response objects."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            return obj.model_dump()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return repr(obj)


def call_vlm(client: OpenAI, model: str, system_text: str, user_text: str,
             image_paths: List[str], temperature: float, max_tokens: int) -> Tuple[str, dict]:
    content: List[dict] = []
    for p in image_paths:
        content.append({"type": "image_url", "image_url": {"url": image_to_b64_url(p)}})
    content.append({"type": "text", "text": user_text})
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    response_debug = to_jsonable(response)
    if not response.choices:
        return "", {"response": response_debug, "client_note": "response contained no choices"}
    message = response.choices[0].message
    text = (message.content or "").strip()
    return text, {
        "response": response_debug,
        "content_type": type(message.content).__name__,
        "content_length": len(message.content or ""),
    }


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*([A-Z])\s*</answer>", re.IGNORECASE)
PLAIN_REASONING_FINAL_RE = re.compile(
    r"(?is)^\s*Reasoning\s*:\s*(.*?)\s*(?:\n|\r\n?)\s*Final\s*:\s*([A-Z])\s*[\.\!\s]*$"
)
PLAIN_FINAL_RE = re.compile(r"(?is)\bFinal\s*:\s*([A-Z])\s*[\.\!\s]*$")
TOOL_CALL_RE = re.compile(r"</?tool_call>", re.IGNORECASE)
ANY_TAG_RE = re.compile(r"</?(?:think|answer|tool_call)>", re.IGNORECASE)
ANSWER_SENTENCE_RE = re.compile(
    r"(?is)(?:^|\n|\.\s*)"
    r"(?:therefore,?\s+|so,?\s+|thus,?\s+|hence,?\s+)?"
    r"(?:the\s+)?(?:final\s+)?(?:correct\s+)?(?:answer|option|choice)"
    r"(?:\s+is|\s*:)?\s+([A-Z])"
    r"(?:[\.\!\s]*$)"
)
ANSWER_TAIL_RES = [
    re.compile(r"(?is)(?:answer|option|choice)\s*(?:is|:)?\s+([A-Z])[\.\!\s]*$"),
    re.compile(r"(?is)\b(?:is|as)\s+(?:option|choice)\s+([A-Z])[\.\!\s]*$"),
    re.compile(r"(?is)(?:option|choice)\s+([A-Z])\s+(?:matches|fits|corresponds\b).*?[\.\!\s]*$"),
    re.compile(r"(?is)\b(?:front|behind|left|right|front-left|front-right|behind-left|behind-right)\s+is\s+([A-Z])[\.\!\s]*$"),
    re.compile(r"(?is)[\-—]\s*(?:option|choice)\s+([A-Z])[\.\!\s]*$"),
]


def parse_output(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Returns (think_text, answer_letter) — either may be None on parse failure."""
    think_m = THINK_RE.search(text)
    ans_m = ANSWER_RE.search(text)
    return (
        think_m.group(1).strip() if think_m else None,
        ans_m.group(1).upper() if ans_m else None,
    )


def normalize_output(text: str, gt_letter: Optional[str] = None,
                     min_reasoning_chars: int = 20) -> dict:
    """Recover a strict training-format answer from common VLM formatting drift.

    Returns a dict containing ``think_text``, ``answer_letter``,
    ``normalized_output``, ``normalization_status``, and ``normalization_notes``.
    ``normalization_status`` is ``"ok"`` only when the answer matches
    ``gt_letter`` (if provided) and the recovered reasoning is non-empty.
    """
    raw = text or ""
    notes: List[str] = []

    if not raw.strip():
        return {
            "think_text": None,
            "answer_letter": None,
            "normalized_output": None,
            "normalization_status": "failed",
            "normalization_notes": ["empty_output"],
        }

    plain_m = PLAIN_REASONING_FINAL_RE.search(raw)
    think_m = THINK_RE.search(raw)
    answer_matches = list(ANSWER_RE.finditer(raw))
    plain_final_m = PLAIN_FINAL_RE.search(raw)
    answer = None
    if plain_m:
        answer = plain_m.group(2).upper()
        notes.append("answer_from_plain_final")
    elif plain_final_m:
        answer = plain_final_m.group(1).upper()
        notes.append("answer_from_plain_final")
    elif answer_matches:
        answer = answer_matches[-1].group(1).upper()
        notes.append("answer_from_tag")
    if answer:
        pass

    if plain_m:
        reasoning = plain_m.group(1)
        notes.append("reasoning_from_plain_reasoning_block")
    elif think_m:
        reasoning = think_m.group(1)
        notes.append("reasoning_from_think_tag")
    else:
        reasoning = raw
        notes.append("reasoning_from_plain_text")

    if TOOL_CALL_RE.search(reasoning):
        notes.append("stripped_tool_call_tags")
    reasoning = TOOL_CALL_RE.sub("", reasoning)

    if plain_m:
        pass
    elif plain_final_m:
        reasoning = reasoning[:plain_final_m.start()].rstrip()
        reasoning = re.sub(r"(?is)^\s*Reasoning\s*:\s*", "", reasoning).strip()
        notes.append("stripped_plain_final_from_reasoning")
    elif answer_matches:
        reasoning = ANSWER_RE.sub("", reasoning)
        notes.append("stripped_answer_tags_from_reasoning")
    else:
        sentence_m = ANSWER_SENTENCE_RE.search(reasoning)
        if sentence_m is None:
            tail_start = max(0, len(reasoning) - 300)
            tail = reasoning[tail_start:]
            for tail_re in ANSWER_TAIL_RES:
                tail_m = tail_re.search(tail)
                if tail_m:
                    sentence_m = tail_m
                    sentence_start = tail_start + tail_m.start()
                    break
            else:
                sentence_start = None
        else:
            sentence_start = sentence_m.start()
        if sentence_m:
            answer = sentence_m.group(1).upper()
            reasoning = reasoning[:sentence_start].rstrip()
            notes.append("answer_from_final_sentence")
            notes.append("stripped_final_answer_sentence")

    # Remove stray schema tags without deleting the content around them.
    if ANY_TAG_RE.search(reasoning):
        notes.append("stripped_stray_schema_tags")
    reasoning = ANY_TAG_RE.sub("", reasoning)

    # Trim common response-role boilerplate if it leaks into content.
    reasoning = re.sub(r"(?im)^\s*(?:assistant|user)\s*:\s*", "", reasoning)
    reasoning = reasoning.strip()

    if gt_letter is not None and answer != gt_letter:
        return {
            "think_text": reasoning or None,
            "answer_letter": answer,
            "normalized_output": None,
            "normalization_status": "failed",
            "normalization_notes": notes + [f"answer_mismatch_gt_{gt_letter}"],
        }

    if len(reasoning) < min_reasoning_chars:
        return {
            "think_text": reasoning or None,
            "answer_letter": answer,
            "normalized_output": None,
            "normalization_status": "failed",
            "normalization_notes": notes + ["reasoning_too_short_or_empty"],
        }

    if answer is None:
        return {
            "think_text": reasoning,
            "answer_letter": None,
            "normalized_output": None,
            "normalization_status": "failed",
            "normalization_notes": notes + ["missing_answer"],
        }

    normalized = f"<think>{reasoning}</think>\n<answer>{answer}</answer>"
    return {
        "think_text": reasoning,
        "answer_letter": answer,
        "normalized_output": normalized,
        "normalization_status": "ok",
        "normalization_notes": notes,
    }


def is_empty_generation(text: str, norm: dict) -> bool:
    return (not (text or "").strip()) or (
        norm.get("normalization_notes") == ["empty_output"]
    )


def build_strict_retry_prompt(user_text: str, gt_letter: str) -> str:
    return (
        user_text
        + f"\n\nIMPORTANT: your Final line MUST contain exactly '{gt_letter}'. "
        "Do not use XML tags. Do not use tool calls. Re-read the metadata, then "
        "output exactly:\n"
        "Reasoning:\n"
        "3-6 sentences of visual reasoning\n"
        f"Final: {gt_letter}"
    )


def build_empty_repair_prompt(user_text: str, gt_letter: str) -> str:
    return (
        user_text
        + "\n\nFORMAT REPAIR FOR EMPTY RESPONSE:\n"
        "Your previous response was empty. Do not use XML tags. Do not use "
        "tool calls. Start your response immediately with Reasoning:. Write "
        "3-6 concise sentences of visual reasoning. Then end with exactly "
        f"Final: {gt_letter}. The final answer letter must be {gt_letter}."
    )


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------

class Annotator:
    def __init__(self, args, metadata_lookup, scene_cache, client, model_name):
        self.args = args
        self.lookup = metadata_lookup
        self.scenes = scene_cache
        self.client = client
        self.model = model_name

    def category_for_id(self, sample_id: str) -> str:
        return category_for_id(sample_id)

    def get_meta(self, sample_id: str, source: str) -> Optional[dict]:
        cat = self.category_for_id(sample_id)
        if cat not in self.lookup:
            return None
        return self.lookup[cat].get(source, {}).get(strip_source_prefix(sample_id))

    def annotate_row(self, row: dict) -> Optional[dict]:
        sample_id = row["id"]
        category = self.category_for_id(sample_id)
        if category not in DISPATCH:
            return self._fail(row, f"unknown category {category}")

        # Resolve which source via image path.
        rel_img1 = row["image"][0]
        source, version_folder, room_part, scene_id = parse_source_from_image_path(rel_img1)
        if source is None:
            return self._fail(row, f"cannot parse source from path {rel_img1}")

        meta = self.get_meta(sample_id, source)
        if meta is None:
            return self._fail(row, f"sample_id {sample_id} not found in {source}")

        # Pull the GT letter from the metadata (more reliable than the jsonl gpt field).
        gt_idx = meta.get("user_1_gt_answer_idx")
        options = meta.get("options_user_1")
        if gt_idx is None or options is None:
            gt_idx = meta.get("user_2_gt_answer_idx")
            options = meta.get("options_user_2")
        if gt_idx is None or options is None:
            return self._fail(row, f"{sample_id}: no GT idx / options")
        gt_letter = chr(65 + gt_idx)

        # Always use question_both_views — it matches the question text the SFT
        # builders place in the training conversation, so the reasoning trace is
        # generated for the same question the model is trained to answer.
        question = meta.get("question_both_views")
        if not question or not str(question).strip():
            return self._fail(row, f"{sample_id}: missing question_both_views")
        question = str(question).strip()

        scene = self.scenes.get(source, version_folder, room_part, scene_id) if source else {}

        module = DISPATCH[category]
        user_text = module.build_prompt(meta, scene, question, options, gt_letter)

        img1_abs = absolute_image_path(row["image"][0])
        img2_abs = absolute_image_path(row["image"][1])
        if not (os.path.exists(img1_abs) and os.path.exists(img2_abs)):
            return self._fail(row, f"image missing: {img1_abs} / {img2_abs}")

        # First attempt.
        out_text, first_debug = self._call_with_retry(SHARED_SYSTEM, user_text, [img1_abs, img2_abs])
        first_norm = normalize_output(out_text, gt_letter)
        final_text = out_text
        final_norm = first_norm
        retry_debug = None
        retry_norm = None
        empty_repair_debug = None
        empty_repair_norm = None

        # Retry once only if the first response cannot be normalized into a
        # correct answer plus non-empty reasoning.
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

        # If the strict retry also collapses into an empty/tool-call-only answer,
        # give the model one deterministic repair attempt with a simpler prompt.
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

        new_gpt = final_norm["normalized_output"]
        new_row = dict(row)
        new_row["conversations"] = [
            row["conversations"][0],
            {"from": "gpt", "value": new_gpt},
        ]
        new_row["_category"] = category
        new_row["_status"] = "ok"
        return new_row

    def _call_with_retry(self, system_text: str, user_text: str, image_paths: List[str],
                         temperature: Optional[float] = None) -> Tuple[str, dict]:
        """Network-level retry only — does NOT retry on bad parse."""
        last_err = None
        request_temperature = self.args.temperature if temperature is None else temperature
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
            "_category": self.category_for_id(row["id"]),
            "_status": "failed",
            "_reason": reason,
            "_raw_output": raw,
        }
        if response_debug is not None:
            failed["_response_debug"] = to_jsonable(response_debug)
        return failed


# ---------------------------------------------------------------------------
# JSONL output (per-category, append-only with a thread lock)
# ---------------------------------------------------------------------------

class CategoryWriter:
    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.locks: Dict[str, threading.Lock] = {}
        self.global_lock = threading.Lock()

    def _path(self, category: str, status: str) -> str:
        if status == "ok":
            return os.path.join(self.output_dir, f"{category}.jsonl")
        return os.path.join(self.output_dir, f"{category}_failed.jsonl")

    def _lock(self, key: str) -> threading.Lock:
        with self.global_lock:
            if key not in self.locks:
                self.locks[key] = threading.Lock()
            return self.locks[key]

    def write(self, result: dict):
        category = result["_category"]
        status = result["_status"]
        path = self._path(category, status)
        clean = {k: v for k, v in result.items() if not k.startswith("_")} if status == "ok" else result
        line = json.dumps(clean) + "\n"
        with self._lock(path):
            with open(path, "a") as f:
                f.write(line)

    def collect_completed_ids(self) -> Dict[str, set]:
        """Returns {category: set(ids)} from existing .jsonl files (resume)."""
        done: Dict[str, set] = {}
        for fname in os.listdir(self.output_dir) if os.path.exists(self.output_dir) else []:
            if not fname.endswith(".jsonl") or fname.endswith("_failed.jsonl"):
                continue
            cat = fname[: -len(".jsonl")]
            ids = done.setdefault(cat, set())
            with open(os.path.join(self.output_dir, fname)) as f:
                for ln in f:
                    try:
                        ids.add(json.loads(ln)["id"])
                    except Exception:
                        pass
        return done


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------

def shard_rows(rows: List[dict], shard_spec: Optional[str]) -> Tuple[List[dict], int, int]:
    if not shard_spec:
        return rows, 0, 1
    idx, total = map(int, shard_spec.split("/"))
    rows_sorted = sorted(rows, key=lambda r: (r["id"], r["image"][0]))
    n = len(rows_sorted)
    size = n // total
    rem = n % total
    start = idx * size + min(idx, rem)
    end = start + size + (1 if idx < rem else 0)
    return rows_sorted[start:end], idx, total


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
    parser.add_argument("--empty_retry_temperature", type=float, default=0.0,
                        help="Temperature for the special retry used after an "
                        "empty model response")
    parser.add_argument("--max_tokens", type=int, default=768)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--network_retries", type=int, default=4)
    parser.add_argument("--shard", default=None, help="e.g. '0/8'")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap rows for smoke testing")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--dump_prompt_for", default=None,
                        help="If set, print the prompt for the first sample of this "
                        "category id-prefix and exit (no LLM call)")
    args = parser.parse_args()

    print(f"Loading dataset metadata...")
    metadata_lookup = build_metadata_lookup()
    scene_cache = SceneCache()

    # Dump-prompt debug mode (no vLLM needed).
    if args.dump_prompt_for:
        dump_prompt(args, metadata_lookup, scene_cache)
        return

    # Connect to vLLM.
    client = OpenAI(base_url=args.api_base, api_key="EMPTY")
    model_name = args.model_name
    if model_name is None:
        model_name = client.models.list().data[0].id
        print(f"Auto-detected model: {model_name}")

    # Load + shard input rows.
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

    # Resume.
    writer = CategoryWriter(args.output_dir)
    if not args.no_resume:
        done = writer.collect_completed_ids()
        if done:
            total_done = sum(len(v) for v in done.values())
            print(f"Resuming: {total_done} rows already in output")
            rows = [
                r for r in rows
                if r["id"] not in done.get(category_for_id(r["id"]), set())
            ]
            print(f"After resume filter: {len(rows)} rows to process")

    annotator = Annotator(args, metadata_lookup, scene_cache, client, model_name)

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


def dump_prompt(args, metadata_lookup, scene_cache):
    """Pure-text smoke test: build the prompt for the first sample of the requested
    category and print it. Useful before launching the vLLM server."""
    cat = args.dump_prompt_for
    if cat not in DISPATCH:
        print(f"Unknown category: {cat}; valid: {list(DISPATCH)}")
        return
    with open(args.input) as f:
        for ln in f:
            row = json.loads(ln)
            if category_for_id(row["id"]) != cat:
                continue
            source, version_folder, room_part, scene_id = parse_source_from_image_path(row["image"][0])
            if source is None:
                continue
            meta = metadata_lookup.get(cat, {}).get(source, {}).get(strip_source_prefix(row["id"]))
            if meta is None:
                continue
            scene = scene_cache.get(source, version_folder, room_part, scene_id)
            options = meta.get("options_user_1") or meta.get("options_user_2") or []
            gt_idx = meta.get("user_1_gt_answer_idx")
            if gt_idx is None:
                gt_idx = meta.get("user_2_gt_answer_idx")
            gt_letter = chr(65 + gt_idx)
            question = meta.get("question_both_views")
            if not question or not str(question).strip():
                raise ValueError(f"{row['id']}: missing question_both_views in metadata")
            question = str(question).strip()
            user_text = DISPATCH[cat].build_prompt(meta, scene, question, options, gt_letter)
            print("=" * 80)
            print(f"id: {row['id']}  source: {source}  room_part: {room_part}  scene_id: {scene_id}")
            print(f"GT: {gt_letter}")
            print("=" * 80)
            print("SYSTEM:")
            print(SHARED_SYSTEM)
            print()
            print("USER:")
            print(user_text)
            return
    print(f"No row found for category {cat}")


if __name__ == "__main__":
    main()
