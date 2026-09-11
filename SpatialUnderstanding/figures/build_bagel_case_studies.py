#!/usr/bin/env python3
"""
Build a single self-contained HTML gallery of BAGEL qualitative case studies.

Layout: bridge (columns: panoramic / point_matching / corner_view)
        × benchmark (rows: COSMIC anchor/counting/rel-distance/rel-direction/map,
                          MMSI, MindCube, OmniSpatial, STARE, BLINK).

Each cell shows N_correct + N_incorrect examples drawn from the corresponding
eval-output directory. For each example: question, options, gold answer,
BAGEL's <think> text (if present), BAGEL's generated reasoning image (inlined
base64), predicted answer, and ✓/✗ status.

Usage:
    python SpatialUnderstanding/figures/build_bagel_case_studies.py
"""

import argparse
import base64
import glob
import html
import json
import os
import random
import re
import sys
from io import BytesIO


EVAL_ROOT_DEFAULT = "/path/to/scratch/VisualCoT"

# Source-dataset roots used to look up input-view images by sample_id, since
# the inference JSONs only carry generated/saved paths, not inputs.
COSMIC_DATASET_ROOT = "/path/to/scratch/VisualCoT/spatial_collab_dataset"
COSMIC_DATASET_FILES = {
    "anchor":        "approved_mcqs_anchor_normalized.json",
    "counting":      "approved_mcqs_counting_normalized.json",
    "rel-distance":  "approved_mcqs_relative_distance_normalized.json",
    "rel-direction": "approved_mcqs_relative_direction_normalized.json",
    "map":           "approved_dataset_map_questions_normalized.json",
}
MINDCUBE_DATASET = "/path/to/scratch/datasets/MindCube/data/raw/MindCube_tinybench.jsonl"
MINDCUBE_IMAGE_ROOT = "/path/to/scratch/datasets/MindCube/data"
# benchmark → loader-tag, used by load_input_index().
INPUT_INDEX_TAG = {
    "anchor": "cosmic", "counting": "cosmic", "rel-distance": "cosmic",
    "rel-direction": "cosmic", "map": "cosmic",
    "mindcube": "mindcube",
    # mmsi inputs come straight from the eval-row's ImagePaths.
    # omnispatial/stare/blink: parquet/no path index — input rendering skipped.
}

# Bridge → benchmark → (subdir-under-root, shard-glob, schema-tag).
# subdir is relative to --eval_root. schema-tag picks the row-normalizer.
BRIDGE_BENCHMARK_DIRS = {
    "panoramic": {
        "anchor":       ("BAGEL_format_mix_all_balance_visual_only_bridge_masked_lora_7k_mcqs_anchor_normalized",       "inference_results_bagel_shard*.json",   "cosmic"),
        "counting":     ("BAGEL_format_mix_all_balance_visual_only_bridge_masked_lora_7k_mcqs_counting_normalized",     "inference_results_bagel_shard*.json",   "cosmic"),
        "rel-distance": ("BAGEL_format_mix_all_balance_visual_only_bridge_masked_lora_7k_mcqs_relative_distance_normalized",  "inference_results_bagel_shard*.json",   "cosmic"),
        "rel-direction":("BAGEL_format_mix_all_balance_visual_only_bridge_masked_lora_7k_mcqs_relative_direction_normalized", "inference_results_bagel_shard*.json",   "cosmic"),
        "map":          ("BAGEL_format_mix_all_balance_visual_only_bridge_masked_lora_7k_map",                          "inference_results_bagel_map_shard*.json", "cosmic"),
        "mmsi":         ("eval_outputs/BAGEL_format_mix_all_balance_visual_only_bridge_masked_lora_6k_mmsi",            "evaluated_model_results_run_1.json",    "mmsi"),
        "mindcube":     ("eval_outputs/BAGEL_format_mix_all_balance_visual_only_bridge_masked_lora_6k_mindcube",        "inference_results_shard*.json",         "mindcube"),
        "omnispatial":  ("omnispatial/BAGEL_format_training_data_mix_all_balance_visual_only_lora_7k_complex_logic_perspective_taking", "inference_results_bagel_shard*.json", "cosmic"),
        "stare":        ("stare_perspective/BAGEL_format_training_data_mix_all_balance_visual_only_lora_7k",            "inference_results_bagel_shard*.json",   "stare"),
        "blink":        ("eval_outputs/BAGEL_format_mix_all_balance_visual_only_bridge_masked_lora_6k_blink",           "inference_results_shard*.json",         "blink"),
    },
    "point_matching": {
        "anchor":       ("BAGEL_format_training_data_pm_no_rotation_visual_only_lora_7k_mcqs_anchor_normalized",        "inference_results_bagel_shard*.json",   "cosmic"),
        "counting":     ("BAGEL_format_training_data_pm_no_rotation_visual_only_lora_7k_mcqs_counting_normalized",      "inference_results_bagel_shard*.json",   "cosmic"),
        "rel-distance": ("BAGEL_format_training_data_pm_no_rotation_visual_only_lora_7k_mcqs_relative_distance_normalized",  "inference_results_bagel_shard*.json", "cosmic"),
        "rel-direction":("BAGEL_format_training_data_pm_no_rotation_visual_only_lora_7k_mcqs_relative_direction_normalized", "inference_results_bagel_shard*.json", "cosmic"),
        "map":          ("BAGEL_format_training_data_pm_no_rotation_visual_only_lora_7k_map",                           "inference_results_bagel_map_shard*.json", "cosmic"),
        "mmsi":         ("eval_outputs/BAGEL_format_training_data_pm_no_rotation_visual_only_lora_mmsi",                "evaluated_model_results_run_1.json",    "mmsi"),
        "mindcube":     ("eval_outputs/BAGEL_format_training_data_pm_no_rotation_visual_only_lora_mindcube",            "inference_results_shard*.json",         "mindcube"),
        "omnispatial":  ("omnispatial/BAGEL_format_training_data_mix_balance_matterport_point_matching_visual_only_lora_complex_logic_perspective_taking", "inference_results_bagel_shard*.json", "cosmic"),
        "stare":        ("stare_perspective/BAGEL_format_training_data_mix_balance_matterport_point_matching_visual_only_lora", "inference_results_bagel_shard*.json", "stare"),
        # blink: no eval dir on disk → SKIP at resolve time
    },
    "corner_view": {
        "anchor":       ("BAGEL_format_corner_view_visual_only_bridge_masked_lora_7k_mcqs_anchor_normalized",           "inference_results_bagel_shard*.json",   "cosmic"),
        "counting":     ("BAGEL_format_corner_view_visual_only_bridge_masked_lora_7k_mcqs_counting_normalized",         "inference_results_bagel_shard*.json",   "cosmic"),
        "rel-distance": ("BAGEL_format_corner_view_visual_only_bridge_masked_lora_7k_mcqs_relative_distance_normalized","inference_results_bagel_shard*.json",   "cosmic"),
        "rel-direction":("BAGEL_format_corner_view_visual_only_bridge_masked_lora_7k_mcqs_relative_direction_normalized","inference_results_bagel_shard*.json",  "cosmic"),
        # map: no eval dir on disk → SKIP
        "mmsi":         ("eval_outputs/BAGEL_format_corner_view_visual_only_bridge_masked_lora_7k_mmsi",                "evaluated_model_results_run_1.json",    "mmsi"),
        # mindcube: no eval dir → SKIP
        "omnispatial":  ("omnispatial/BAGEL_format_corner_view_visual_only_bridge_masked_lora_7k_complex_logic_perspective_taking", "inference_results_bagel_shard*.json", "cosmic"),
        "stare":        ("stare_perspective/BAGEL_format_corner_view_visual_only_bridge_masked_lora_7k",                "inference_results_bagel_shard*.json",   "stare"),
        # blink: no eval dir → SKIP
    },
}

BENCHMARKS = ["anchor", "counting", "rel-distance", "rel-direction", "mmsi", "mindcube", "omnispatial", "stare", "blink"]
BRIDGES = ["panoramic", "point_matching", "corner_view"]

# COSMIC subtasks used for curated comparisons (drops map per user request).
COSMIC_SUBTASKS = ["anchor", "counting", "rel-distance", "rel-direction"]

# User-named sample_ids to highlight in the curated section, organized by
# benchmark. Verified to exist on disk; their per-bridge correctness is
# checked at render time and rendered as-is (✓/✗) regardless of whether
# they form a clean "panorama-only-wins" pattern.
CURATED_SAMPLES = {
    "anchor":       ["anchor_001147"],
    "counting":     ["counting_003053"],
    "rel-distance": ["relative_distance_014220", "relative_distance_010670"],
}

# Per-subtask filter for the auto-find section. Three filter shapes:
#   - "strict_pano_only" : pano=1, pm=0, cv=0   (panorama is the only winner)
#   - "pano_beats_any"   : pano=1, (pm=0 OR cv=0)   (panorama wins over at least one)
#   - "ids:<sid1>,<sid2>": exact sample_ids (no filtering)
# Cap is applied with max_wins_per_subtask EXCEPT for "ids:" which always
# shows all named IDs.
AUTOFIND_FILTERS = {
    "anchor":        "pano_beats_any",      # widened per user — 14 candidates
    "counting":      "strict_pano_only",    # 12 strict wins
    "rel-distance":  "strict_pano_only",    # 11 strict wins
    "rel-direction": "ids:spatial_008738",  # keep only this one
}

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def remap_symlink_path(path):
    """Resolve the /path/to/scratch/... symlink prefix used in some
    saved_image_paths to the real /path/to/scratch/... path."""
    if not path:
        return path
    return path.replace("/path/to/scratch/", "/path/to/scratch/")


def img_to_data_uri(path, max_dim=768, jpeg_quality=78):
    """Inline a PNG/JPG as a base64 data: URI, downscaled to max_dim on the long side."""
    if not path or not os.path.exists(path):
        return ""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if max(w, h) > max_dim:
                if w >= h:
                    nw, nh = max_dim, int(h * max_dim / w)
                else:
                    nw, nh = int(w * max_dim / h), max_dim
                im = im.resize((nw, nh), Image.LANCZOS)
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=jpeg_quality)
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    except Exception as e:
        print(f"  [IMG-WARN] {path}: {e}")
        return ""


def parse_think_answer(text):
    if not text:
        return "", ""
    think = ""
    m = THINK_RE.search(text)
    if m:
        think = m.group(1).strip()
    ans = ""
    m = ANSWER_RE.search(text)
    if m:
        ans = m.group(1).strip()
    return think, ans


_INPUT_INDEX_CACHE = {}


def load_input_index(benchmark):
    """Return sample_id → list[input_image_path] for the given benchmark, or
    {} if not supported. Cached across cells (3 bridges hit the same index)."""
    if benchmark in _INPUT_INDEX_CACHE:
        return _INPUT_INDEX_CACHE[benchmark]
    tag = INPUT_INDEX_TAG.get(benchmark)
    idx = {}
    if tag == "cosmic":
        fn = COSMIC_DATASET_FILES.get(benchmark)
        path = os.path.join(COSMIC_DATASET_ROOT, fn) if fn else None
        if path and os.path.exists(path):
            for row in json.load(open(path)):
                sid = row.get("sample_id")
                if not sid:
                    continue
                paths = []
                for k in ("user_1_image_local_path", "user_2_image_local_path"):
                    p = row.get(k)
                    if p:
                        paths.append(p)
                # For map questions the proposed top-down being judged is also
                # an input; append it as a third view if present.
                map_img = row.get("map_image_path")
                if map_img:
                    paths.append(map_img)
                idx[sid] = paths
            print(f"  [INPUT-IDX] {benchmark}: {len(idx)} cosmic rows from {fn}")
        else:
            print(f"  [INPUT-IDX] {benchmark}: dataset file missing → {path}")
    elif tag == "mindcube":
        if os.path.exists(MINDCUBE_DATASET):
            with open(MINDCUBE_DATASET) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sid = row.get("id")
                    imgs = row.get("images") or []
                    if sid:
                        idx[sid] = [os.path.join(MINDCUBE_IMAGE_ROOT, p) for p in imgs]
            print(f"  [INPUT-IDX] mindcube: {len(idx)} rows from {MINDCUBE_DATASET}")
        else:
            print(f"  [INPUT-IDX] mindcube: dataset file missing → {MINDCUBE_DATASET}")
    _INPUT_INDEX_CACHE[benchmark] = idx
    return idx


def load_eval_dir(eval_dir, shard_glob):
    """Load all shards under eval_dir matching shard_glob. Returns the
    flattened list of result rows. Each shard JSON is either a dict with
    'results': [...] (most benchmarks) or a top-level list (MMSI).

    Special case for MMSI: prefer `evaluated_model_results_run_1.json` (judged)
    when present; else fall back to per-shard unjudged files (caller will use
    ExtractedAnswer==GroundTruth text match as correctness)."""
    # If shard_glob refers to a single specific file (no wildcard), only load it.
    if "*" not in shard_glob:
        sp = os.path.join(eval_dir, shard_glob)
        if os.path.exists(sp):
            try:
                d = json.load(open(sp))
                return d if isinstance(d, list) else (d.get("results", []) or [])
            except Exception as e:
                print(f"  [LOAD-WARN] {sp}: {e}")
                return []
        # Fall back to MMSI per-shard files when the judged JSON is missing.
        fallback_glob = "model_results_run_1_shard*.json"
        shard_paths = sorted(glob.glob(os.path.join(eval_dir, fallback_glob)))
    else:
        shard_paths = sorted(glob.glob(os.path.join(eval_dir, shard_glob)))
    rows = []
    for sp in shard_paths:
        try:
            d = json.load(open(sp))
        except Exception as e:
            print(f"  [LOAD-WARN] {sp}: {e}")
            continue
        if isinstance(d, dict):
            rows.extend(d.get("results", []) or [])
        elif isinstance(d, list):
            rows.extend(d)
    return rows


def normalize_row(row, schema, eval_dir):
    """Return a dict with normalized fields:
        question (str), options (list[str] or []), gold (str), pred (str),
        is_correct (bool), image_path (str or ''), input_image_paths (list[str]),
        think (str), sample_id (str), extra_meta (str — small label).

    schema ∈ {cosmic, blink, mindcube, stare, mmsi}."""
    out = {
        "question": "", "options": [], "gold": "", "pred": "",
        "is_correct": False, "image_path": "", "input_image_paths": [],
        "think": "", "sample_id": "", "extra_meta": "",
    }
    if schema == "mmsi":
        out["question"] = str(row.get("Question") or row.get("OriginalQuestion") or "")
        out["gold"] = str(row.get("GroundTruth", ""))
        out["pred"] = str(row.get("ExtractedAnswer", ""))
        # LLMJudgeResult is a boolean. Some eval runs left it False on every
        # row (judge not run on the shards); fall back to ExtractedAnswer
        # == GroundTruth string match in that case.
        judged = bool(row.get("LLMJudgeResult"))
        if judged:
            out["is_correct"] = True
        else:
            ea = str(row.get("ExtractedAnswer", "")).strip().upper()
            gt = str(row.get("GroundTruth", "")).strip().upper()
            out["is_correct"] = bool(ea) and ea == gt
        out["think"] = str(row.get("Thought", ""))
        # GeneratedImages is a list of paths; first one is the BAGEL output.
        # The path embedded in the JSON may point to a sibling dir (the eval
        # rerun renamed the output dir). Anchor on the "/pass_1/" component
        # and rebuild relative to the actual eval_dir.
        gen = row.get("GeneratedImages") or []
        if gen:
            p = remap_symlink_path(gen[0])
            if not os.path.exists(p):
                idx = p.find("/pass_1/")
                if idx >= 0:
                    p = os.path.join(eval_dir, p[idx + 1 :])
            out["image_path"] = p if os.path.exists(p) else ""
        out["sample_id"] = str(row.get("Id", ""))
        out["extra_meta"] = str(row.get("QuestionType", ""))
        # Input views: ImagePaths field, anchored on /images/ when the symlink
        # path is stale (eval rerun renamed the parent dir).
        inputs = []
        for p_raw in (row.get("ImagePaths") or []):
            p = remap_symlink_path(p_raw)
            if not os.path.exists(p):
                idx = p.find("/images/")
                if idx >= 0:
                    p = os.path.join(eval_dir, p[idx + 1 :])
            if os.path.exists(p):
                inputs.append(p)
        out["input_image_paths"] = inputs
        return out

    # Common BAGEL output schema: final_answer_text + saved_image_paths.
    final = row.get("final_answer_text", "")
    think, _ans = parse_think_answer(final)
    out["think"] = think
    paths = row.get("saved_image_paths") or []
    if paths:
        p = remap_symlink_path(paths[0])
        if not os.path.exists(p):
            # Fallback: re-anchor on /generated_images/ so the path inside the
            # actual eval_dir is preserved even if upstream eval renamed the
            # parent dir.
            idx = p.find("/generated_images/")
            if idx >= 0:
                p = os.path.join(eval_dir, p[idx + 1 :])
            else:
                p = os.path.join(eval_dir, "generated_images", os.path.basename(paths[0]))
        out["image_path"] = p if os.path.exists(p) else ""
    out["question"] = str(row.get("question", ""))
    out["options"] = list(row.get("options", []) or [])
    out["pred"] = str(row.get("predicted_answer", ""))
    out["is_correct"] = float(row.get("accuracy", 0.0)) >= 0.5
    out["sample_id"] = str(row.get("sample_id", row.get("qid", row.get("_idx", ""))))

    if schema == "cosmic":
        out["gold"] = str(row.get("correct_answer", ""))
        out["extra_meta"] = str(row.get("question_type", row.get("task_type", row.get("sub_task_type", ""))))
    elif schema == "blink":
        out["gold"] = str(row.get("gt_answer", ""))
        out["extra_meta"] = str(row.get("sub_task", ""))
    elif schema == "mindcube":
        out["gold"] = str(row.get("gt_answer", ""))
        cat = row.get("category", "")
        if isinstance(cat, list):
            cat = ", ".join(str(c) for c in cat)
        out["extra_meta"] = str(cat)
    elif schema == "stare":
        out["gold"] = str(row.get("correct_answer", ""))
        out["extra_meta"] = str(row.get("category", ""))
    return out


def sample_rows(eval_dir, shard_glob, schema, benchmark, n_correct, n_incorrect, seed):
    rows = load_eval_dir(eval_dir, shard_glob)
    normed = [normalize_row(r, schema, eval_dir) for r in rows]
    # Drop rows with no resolvable image.
    normed = [r for r in normed if r["image_path"]]
    # Attach input views from the external index where supported (cosmic/mindcube).
    # MMSI already populates input_image_paths from its own ImagePaths field.
    input_idx = load_input_index(benchmark)
    if input_idx:
        for r in normed:
            if not r["input_image_paths"]:
                paths = [p for p in input_idx.get(r["sample_id"], []) if os.path.exists(p)]
                r["input_image_paths"] = paths
    correct = [r for r in normed if r["is_correct"]]
    incorrect = [r for r in normed if not r["is_correct"]]
    rng = random.Random(seed)
    rng.shuffle(correct)
    rng.shuffle(incorrect)
    return correct[:n_correct] + incorrect[:n_incorrect], len(normed), len(correct), len(incorrect)


def find_row_by_sid(eval_dir, shard_glob, schema, sample_id):
    """Linear scan: load all shards, return the first normalized row matching
    `sample_id`. Returns None if not found."""
    rows = load_eval_dir(eval_dir, shard_glob)
    for r in rows:
        # The MMSI schema uses 'Id' (int), all others 'sample_id'.
        rid = r.get("sample_id") if "sample_id" in r else r.get("Id")
        if rid is not None and str(rid) == str(sample_id):
            return normalize_row(r, schema, eval_dir)
    return None


def collect_curated_row(benchmark, sample_id, eval_root):
    """For one (benchmark, sample_id), load the normalized row from each of
    the 3 bridges. Returns dict bridge → normalized-row-or-None. Also attaches
    the input-view paths from the COSMIC dataset index (same for all bridges,
    since inputs are the question's source views)."""
    per_bridge = {}
    for bridge in BRIDGES:
        spec = BRIDGE_BENCHMARK_DIRS.get(bridge, {}).get(benchmark)
        if spec is None:
            per_bridge[bridge] = None
            continue
        subdir, shard_glob, schema = spec
        eval_dir = os.path.join(eval_root, subdir)
        if not os.path.isdir(eval_dir):
            per_bridge[bridge] = None
            continue
        per_bridge[bridge] = find_row_by_sid(eval_dir, shard_glob, schema, sample_id)
    # Attach input views once (same source images regardless of bridge).
    input_idx = load_input_index(benchmark)
    inputs = [p for p in input_idx.get(sample_id, []) if os.path.exists(p)] if input_idx else []
    for r in per_bridge.values():
        if r is not None and not r["input_image_paths"]:
            r["input_image_paths"] = inputs
    return per_bridge


def render_bridge_cell(ex, bridge):
    """Render one bridge's column in a curated-row layout: just the BAGEL
    output + pred + ✓/✗ (question/options/gold/inputs are shared header)."""
    if ex is None:
        return '<td class="bridge-cell skip">— not in this checkpoint —</td>'
    status_cls = "ok" if ex["is_correct"] else "bad"
    status_sym = "✓" if ex["is_correct"] else "✗"
    bagel_uri = img_to_data_uri(ex["image_path"], max_dim=768)
    bagel_html = f'<img src="{bagel_uri}" />' if bagel_uri else '<div class="missing">image missing</div>'
    think_html = ""
    if ex["think"]:
        think_html = f'<details class="think"><summary>think</summary><div class="think-body">{html.escape(ex["think"])}</div></details>'
    return (
        f'<td class="bridge-cell">'
        f'<div class="bridge-status"><span class="status {status_cls}">{status_sym}</span> '
        f'<b>pred:</b> {html.escape(ex["pred"])}</div>'
        f'<div class="bagel-img">{bagel_html}</div>'
        f'{think_html}'
        f'</td>'
    )


def render_curated_row(benchmark, sample_id, per_bridge):
    """Render one curated sample as a single row: header (q/options/inputs/gold)
    spanning all bridges + one cell per bridge with the BAGEL output."""
    # Pick a non-None row to read shared fields from.
    ref = next((r for r in per_bridge.values() if r is not None), None)
    if ref is None:
        return (
            f'<tr><td colspan="{len(BRIDGES)+1}" class="curated-missing">'
            f'<b>{html.escape(benchmark)}/{html.escape(sample_id)}</b>: not found in any checkpoint'
            f'</td></tr>'
        )
    # Options
    if ref["options"]:
        letters = "ABCDEFGHIJ"
        opts_html = "<ul class='opts'>" + "".join(
            f"<li><b>{letters[i] if i < len(letters) else i}.</b> {html.escape(str(o))}</li>"
            for i, o in enumerate(ref["options"])
        ) + "</ul>"
    else:
        opts_html = ""
    # Input thumbnails
    inputs = ref.get("input_image_paths", [])[:4]
    in_cells = []
    for i, p in enumerate(inputs):
        uri = img_to_data_uri(p, max_dim=420, jpeg_quality=72)
        if uri:
            in_cells.append(f'<div class="in-cell"><div class="in-lbl">view {i+1}</div><img src="{uri}" /></div>')
    inputs_html = f'<div class="inputs">{"".join(in_cells)}</div>' if in_cells else ""

    header_td = (
        f'<td class="curated-header">'
        f'<div class="ex-head"><code class="sid">{html.escape(sample_id)}</code>'
        f'<span class="meta-chip">{html.escape(benchmark)}</span></div>'
        f'<div class="qa-block"><div class="qtext"><b>Q:</b> {html.escape(ref["question"])}</div>{opts_html}</div>'
        f'{inputs_html}'
        f'<div class="answers"><span class="gold"><b>gold:</b> {html.escape(ref["gold"])}</span></div>'
        f'</td>'
    )
    bridge_tds = "".join(render_bridge_cell(per_bridge[b], b) for b in BRIDGES)
    return f'<tr class="curated-row">{header_td}{bridge_tds}</tr>'


def find_panorama_wins(eval_root, subtasks, filters):
    """For each subtask, return a list of sample_ids matching its filter.

    `filters[sub]` is one of:
      - "strict_pano_only": pano=1 AND pm=0 AND cv=0
      - "pano_beats_any":   pano=1 AND (pm=0 OR cv=0)
      - "ids:<sid>,<sid>":  literal sample_id list, no filtering
    """
    wins = {}
    for sub in subtasks:
        spec = filters.get(sub, "strict_pano_only")
        if spec.startswith("ids:"):
            ids = [s.strip() for s in spec[4:].split(",") if s.strip()]
            wins[sub] = ids
            print(f"  [WINS] {sub}: {len(ids)} explicit sample_id(s) ({spec})")
            continue
        per_bridge_rows = {}
        for bridge in BRIDGES:
            bspec = BRIDGE_BENCHMARK_DIRS.get(bridge, {}).get(sub)
            if bspec is None:
                per_bridge_rows[bridge] = {}
                continue
            subdir, shard_glob, _schema = bspec
            eval_dir = os.path.join(eval_root, subdir)
            if not os.path.isdir(eval_dir):
                per_bridge_rows[bridge] = {}
                continue
            rows = load_eval_dir(eval_dir, shard_glob)
            per_bridge_rows[bridge] = {r["sample_id"]: float(r.get("accuracy", 0.0)) for r in rows if r.get("sample_id")}
        common = set(per_bridge_rows["panoramic"]) & set(per_bridge_rows["point_matching"]) & set(per_bridge_rows["corner_view"])
        if spec == "strict_pano_only":
            sub_wins = sorted([
                s for s in common
                if per_bridge_rows["panoramic"][s] == 1.0
                and per_bridge_rows["point_matching"][s] == 0.0
                and per_bridge_rows["corner_view"][s] == 0.0
            ])
        elif spec == "pano_beats_any":
            sub_wins = sorted([
                s for s in common
                if per_bridge_rows["panoramic"][s] == 1.0
                and (per_bridge_rows["point_matching"][s] == 0.0
                     or per_bridge_rows["corner_view"][s] == 0.0)
            ])
        else:
            raise ValueError(f"unknown filter spec for {sub}: {spec!r}")
        wins[sub] = sub_wins
        print(f"  [WINS] {sub}: {len(sub_wins)} sample_ids matching {spec!r}")
    return wins


def render_example(ex, bridge, benchmark, idx_in_cell):
    """Render one example as an HTML block.

    Layout (top → bottom):
      header chip · sample_id
      question (bold)
      options (if any)
      input views (if resolved)  ← cam0 | cam1 | …
      gold / pred (✓/✗)
      generated reasoning image (BAGEL output)
      [think] expandable
    """
    status_cls = "ok" if ex["is_correct"] else "bad"
    status_sym = "✓" if ex["is_correct"] else "✗"

    bagel_uri = img_to_data_uri(ex["image_path"], max_dim=768)
    bagel_html = f'<img src="{bagel_uri}" />' if bagel_uri else '<div class="missing">image missing</div>'

    # Options list (numbered A/B/C/...) — only if options non-empty.
    if ex["options"]:
        letters = "ABCDEFGHIJ"
        opts_html = "<ul class='opts'>" + "".join(
            f"<li><b>{letters[i] if i < len(letters) else i}.</b> {html.escape(str(o))}</li>"
            for i, o in enumerate(ex["options"])
        ) + "</ul>"
    else:
        opts_html = ""

    # Input views — render up to 4 thumbnails side by side.
    inputs = ex.get("input_image_paths", [])[:4]
    if inputs:
        cells = []
        for i, p in enumerate(inputs):
            uri = img_to_data_uri(p, max_dim=420, jpeg_quality=72)
            if uri:
                cells.append(f'<div class="in-cell"><div class="in-lbl">view {i+1}</div><img src="{uri}" /></div>')
        inputs_html = f'<div class="inputs">{"".join(cells)}</div>' if cells else ""
    else:
        inputs_html = ""

    think_html = ""
    if ex["think"]:
        t = html.escape(ex["think"])
        think_html = f'<details class="think"><summary>think</summary><div class="think-body">{t}</div></details>'

    meta_chip = f'<span class="meta-chip">{html.escape(ex["extra_meta"])}</span>' if ex["extra_meta"] else ""
    sid_html = f'<code class="sid">{html.escape(ex["sample_id"])}</code>'

    return f"""
<div class="ex">
  <div class="ex-head">
    <span class="ex-idx">#{idx_in_cell}</span>
    <span class="status {status_cls}">{status_sym}</span>
    {meta_chip}
    {sid_html}
  </div>
  <div class="qa-block">
    <div class="qtext"><b>Q:</b> {html.escape(ex["question"])}</div>
    {opts_html}
  </div>
  {inputs_html}
  <div class="answers">
    <span class="gold"><b>gold:</b> {html.escape(ex["gold"])}</span>
    <span class="pred {status_cls}"><b>pred:</b> {html.escape(ex["pred"])}</span>
  </div>
  <div class="bagel-img"><div class="bagel-lbl">BAGEL generated reasoning</div>{bagel_html}</div>
  {think_html}
</div>
"""


HEAD = """<!doctype html>
<html><head><meta charset="utf-8"><title>BAGEL Case Studies — bridge × benchmark</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 16px; background: #fafafa; color: #222; }
  h1 { margin: 0 0 6px 0; }
  .summary { color: #555; margin-bottom: 16px; font-size: 13px; }
  table.grid { border-collapse: collapse; width: 100%; }
  table.grid th, table.grid td { border: 1px solid #ccc; vertical-align: top; padding: 0; }
  table.grid thead th { background: #2c3e50; color: #fff; padding: 8px 10px; position: sticky; top: 0; z-index: 3; }
  table.grid tbody th.bench { background: #ecf0f1; padding: 10px; writing-mode: horizontal-tb; position: sticky; left: 0; z-index: 2; min-width: 130px; max-width: 130px; font-size: 14px; }
  td.cell { padding: 8px; min-width: 320px; max-width: 380px; background: #fff; }
  td.cell.skip { background: #f4f4f4; color: #888; text-align: center; font-style: italic; padding: 24px; }
  .cell-head { font-size: 11px; color: #777; margin-bottom: 6px; }
  .ex { border: 1px solid #ddd; border-radius: 6px; padding: 8px; margin-bottom: 10px; background: #fff; }
  .ex-head { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 6px; font-size: 12px; }
  .ex-idx { color: #888; font-weight: 700; }
  .status { font-size: 14px; font-weight: 700; padding: 0 6px; border-radius: 3px; }
  .status.ok { background: #d6f5d6; color: #14532d; }
  .status.bad { background: #fde2e2; color: #7f1d1d; }
  .meta-chip { background: #eef; color: #336; padding: 1px 6px; border-radius: 3px; font-size: 11px; }
  code.sid { font-size: 10px; color: #888; background: #f5f5f5; padding: 1px 4px; border-radius: 2px; }
  .qa-block { background: #f8fafc; border-left: 3px solid #3b82f6; padding: 6px 8px; margin: 4px 0; }
  .qtext { font-size: 12px; color: #222; margin-bottom: 4px; line-height: 1.35; }
  .qtext b { color: #1e40af; }
  ul.opts { margin: 2px 0 0 4px; padding: 0; font-size: 11px; color: #444; }
  ul.opts li { margin: 1px 0; list-style: none; }
  .inputs { display: flex; gap: 6px; margin: 6px 0 4px 0; flex-wrap: wrap; }
  .in-cell { display: flex; flex-direction: column; align-items: center; flex: 1 1 0; min-width: 60px; }
  .in-cell img { max-width: 100%; max-height: 140px; border: 1px solid #ccc; border-radius: 3px; }
  .in-lbl { font-size: 9px; color: #777; margin-bottom: 2px; }
  .answers { font-size: 12px; margin: 6px 0 4px 0; display: flex; gap: 12px; flex-wrap: wrap; padding: 4px 6px; background: #fefce8; border-radius: 3px; }
  .gold { color: #14532d; }
  .pred.ok { color: #14532d; }
  .pred.bad { color: #7f1d1d; }
  details.think { margin: 4px 0; font-size: 11px; }
  details.think summary { cursor: pointer; color: #555; }
  .think-body { background: #fafafa; border-left: 3px solid #ccc; padding: 4px 8px; margin: 4px 0; max-height: 200px; overflow-y: auto; white-space: pre-wrap; font-size: 11px; }
  .bagel-lbl { font-size: 10px; color: #6b21a8; font-weight: 600; text-align: center; margin-top: 4px; }
  .bagel-img img { max-width: 100%; max-height: 280px; border: 2px solid #a78bfa; border-radius: 4px; display: block; margin: 4px auto 0; }
  .missing { width: 100%; height: 80px; display: flex; align-items: center; justify-content: center; border: 1px dashed #f00; color: #f00; font-size: 11px; }
  /* Curated mode (sample-id rows × bridge columns) */
  h2 { margin: 24px 0 8px 0; padding-bottom: 4px; border-bottom: 2px solid #2c3e50; }
  table.curated { border-collapse: collapse; width: 100%; margin-bottom: 32px; }
  table.curated thead th { background: #2c3e50; color: #fff; padding: 8px 10px; position: sticky; top: 0; z-index: 3; border: 1px solid #1a252f; }
  table.curated thead th.header-col { min-width: 260px; max-width: 320px; }
  table.curated thead th.bridge-col { min-width: 320px; }
  table.curated td { border: 1px solid #ccc; vertical-align: top; padding: 10px; background: #fff; }
  td.curated-header { background: #f8fafc; min-width: 260px; max-width: 320px; }
  td.bridge-cell { min-width: 320px; max-width: 420px; }
  td.bridge-cell.skip { background: #f4f4f4; color: #888; text-align: center; font-style: italic; padding: 24px; }
  .bridge-status { font-size: 13px; margin-bottom: 6px; padding: 4px 6px; background: #fefce8; border-radius: 3px; }
  .curated-missing { background: #fde2e2; color: #7f1d1d; padding: 16px; text-align: center; font-style: italic; }
  tr.curated-row + tr.curated-row td { border-top: 2px solid #888; }
  .subtask-heading { background: #ecf0f1; padding: 6px 12px; border-left: 4px solid #2c3e50; margin: 16px 0 6px 0; font-weight: 700; font-size: 14px; }
</style>
</head><body>
"""


def build_matrix_section(args):
    """Original bridge × benchmark matrix. Returns list of HTML parts."""
    cell_data = {}
    n_resolved = n_skipped = 0
    for bench in BENCHMARKS:
        for bridge in BRIDGES:
            spec = BRIDGE_BENCHMARK_DIRS.get(bridge, {}).get(bench)
            if spec is None:
                print(f"  [SKIP] {bridge:<14}/{bench:<14}: not configured")
                cell_data[(bridge, bench)] = (None, None)
                n_skipped += 1
                continue
            subdir, shard_glob, schema = spec
            eval_dir = os.path.join(args.eval_root, subdir)
            if not os.path.isdir(eval_dir):
                print(f"  [SKIP] {bridge:<14}/{bench:<14}: dir missing: {eval_dir}")
                cell_data[(bridge, bench)] = (None, None)
                n_skipped += 1
                continue
            cell_seed = args.seed + abs(hash((bridge, bench))) % (2**31)
            examples, n_total, n_ok, n_bad = sample_rows(
                eval_dir, shard_glob, schema, bench,
                args.n_correct, args.n_incorrect, cell_seed,
            )
            print(f"  [OK]   {bridge:<14}/{bench:<14}: {len(examples)} ex (from {n_total} rows, {n_ok}✓/{n_bad}✗) — {subdir}")
            cell_data[(bridge, bench)] = (examples, eval_dir)
            n_resolved += 1
    print(f"  Resolved {n_resolved} cells · skipped {n_skipped}")

    parts = []
    parts.append("<h2>Random samples — bridge × benchmark</h2>")
    parts.append(
        f'<div class="summary">Each cell: up to {args.n_correct} correct + {args.n_incorrect} incorrect '
        f'examples sampled from the matching eval-output dir. ✓ = predicted matches gold. '
        f'Image shown is the reasoning bridge BAGEL generated at inference time. seed={args.seed}.</div>'
    )
    parts.append('<table class="grid"><thead><tr><th></th>')
    for bridge in BRIDGES:
        parts.append(f"<th>{bridge}</th>")
    parts.append("</tr></thead><tbody>")
    for bench in BENCHMARKS:
        parts.append(f'<tr><th class="bench">{bench}</th>')
        for bridge in BRIDGES:
            examples, _ = cell_data[(bridge, bench)]
            if examples is None:
                parts.append('<td class="cell skip">— not evaluated —</td>')
                continue
            if not examples:
                parts.append('<td class="cell skip">— no usable rows —</td>')
                continue
            cell_parts = [f'<td class="cell"><div class="cell-head">{len(examples)} ex</div>']
            for i, ex in enumerate(examples):
                cell_parts.append(render_example(ex, bridge, bench, i))
            cell_parts.append("</td>")
            parts.append("".join(cell_parts))
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return parts


def build_curated_table(title, sample_groups, eval_root):
    """Render a curated-table section. `sample_groups` is a list of
    (subtask, [sample_id, ...]) tuples (preserves order)."""
    parts = []
    parts.append(f"<h2>{html.escape(title)}</h2>")
    parts.append('<table class="curated"><thead><tr>')
    parts.append('<th class="header-col">Question · inputs · gold</th>')
    for b in BRIDGES:
        parts.append(f'<th class="bridge-col">{b}</th>')
    parts.append('</tr></thead><tbody>')
    for subtask, sids in sample_groups:
        if not sids:
            continue
        parts.append(
            f'<tr><td colspan="{len(BRIDGES)+1}"><div class="subtask-heading">{html.escape(subtask)} '
            f'<span style="color:#666;font-weight:400;font-size:12px;">· {len(sids)} sample(s)</span></div></td></tr>'
        )
        for sid in sids:
            per_bridge = collect_curated_row(subtask, sid, eval_root)
            parts.append(render_curated_row(subtask, sid, per_bridge))
    parts.append("</tbody></table>")
    return parts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_root", default=EVAL_ROOT_DEFAULT)
    p.add_argument("--mode", choices=["matrix", "curated", "both"], default="curated",
                   help="matrix=original bridge×benchmark random sampling. "
                        "curated=user-named sample_ids + auto-found panorama-only wins. "
                        "both=concatenate.")
    p.add_argument("--n_correct", type=int, default=2)
    p.add_argument("--n_incorrect", type=int, default=2)
    p.add_argument("--max_wins_per_subtask", type=int, default=4,
                   help="In curated mode: cap on auto-found panorama-only-wins shown per subtask.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_html", default=os.path.expanduser("~/scratch/VisualCoT/case_studies/bagel_case_studies.html"))
    args = p.parse_args()

    print(f"Building gallery (mode={args.mode}) → {args.output_html}")

    parts = [HEAD]
    parts.append(f"<h1>BAGEL Case Studies</h1>")

    if args.mode in ("curated", "both"):
        # Section 1: user-named sample_ids.
        print("\n=== Section 1: user-curated sample_ids ===")
        curated_groups = [(sub, sids) for sub, sids in CURATED_SAMPLES.items()]
        parts.extend(build_curated_table(
            "User-curated sample_ids (panorama vs. point-matching vs. corner-view)",
            curated_groups, args.eval_root,
        ))

        # Section 2: auto-found panorama wins (per-subtask filter spec).
        print("\n=== Section 2: auto-found panorama wins ===")
        wins = find_panorama_wins(args.eval_root, COSMIC_SUBTASKS, AUTOFIND_FILTERS)
        # Deterministic-shuffle each subtask's wins and cap, EXCEPT when the
        # filter is an explicit "ids:..." list (always show all named IDs)
        # or when --no_cap is on for a subtask flagged anchor=all (special-cased
        # via the "all" sentinel — we simply use a very large cap).
        rng = random.Random(args.seed)
        win_groups = []
        for sub in COSMIC_SUBTASKS:
            sids = list(wins.get(sub, []))
            filter_spec = AUTOFIND_FILTERS.get(sub, "strict_pano_only")
            if filter_spec.startswith("ids:"):
                pass  # keep order as-provided
            else:
                rng.shuffle(sids)
            # Anchor with "pano_beats_any" shows all 14 (per user instruction);
            # other subtasks keep the per-CLI cap.
            if sub == "anchor" and filter_spec == "pano_beats_any":
                capped = sids
            elif filter_spec.startswith("ids:"):
                capped = sids
            else:
                capped = sids[: args.max_wins_per_subtask]
            win_groups.append((sub, capped))
        parts.extend(build_curated_table(
            "Auto-discovered panorama wins "
            "(anchor: panoramic ✓ AND (pm ✗ OR cv ✗); "
            "counting & rel-distance: strict panoramic-only wins; "
            "rel-direction: spatial_008738 only)",
            win_groups, args.eval_root,
        ))

    if args.mode in ("matrix", "both"):
        print("\n=== Random samples — bridge × benchmark matrix ===")
        parts.extend(build_matrix_section(args))

    parts.append("</body></html>")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_html)), exist_ok=True)
    with open(args.output_html, "w") as f:
        f.write("".join(parts))
    sz = os.path.getsize(args.output_html) / 1e6
    print(f"\nWrote {args.output_html} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
