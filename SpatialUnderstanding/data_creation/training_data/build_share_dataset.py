#!/usr/bin/env python3
"""
Build a self-contained shareable dataset bundle from the four training variants.

Output layout (default root: /path/to/scratch/infinigen/share_dataset/):

    share_dataset/
      images/<source>/<scene_id>/
        cam0.png            # copy of Image_0_0_0048_0.png
        cam1.png            # copy of Image_1_0_0048_0.png
        panorama.png        # copy of panorama_blender_limits_*.png (if scene has one)
        topdown_blender.png # copy of topdown_blender.png            (if scene has one)
        pm_bridge.png       # composited from annotated_cam0+1.png  (if scene has PM anno)
      no_thinking.jsonl
      visual_only_panorama.jsonl
      visual_only_topdown.jsonl
      visual_only_pm.jsonl

All four JSONLs follow the no_thinking schema: {conversations, image, id}.
Visual-only variants have 3 image paths and a `<image_start><image_end><answer>X</answer>`
gpt response. No-thinking has 2 image paths and just `<answer>X</answer>`.

Intersection-first: only ids present in all 4 variants survive. The script
prints id-set sizes + intersection size before doing any image copying.

Usage:
    # dry-run (pass 1 only — print intersection size, no writes)
    python build_share_dataset.py --dry_run

    # full build
    python build_share_dataset.py
"""

import argparse
import io
import json
import os
import random
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from PIL import Image
from tqdm import tqdm


# ─── Constants from the source data-creation scripts ────────────────────────────

NO_THINKING_SYSTEM_PROMPT = (
    "Answer the question directly. "
    "Provide your final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>."
)
VISUAL_ONLY_THINK_SYSTEM_PROMPT = (
    "Think visually to answer the question. "
    "Enclose your visual thinking within <image_start> </image_end>."
)

DEFAULT_NO_THINKING_JSONL = (
    "/path/to/scratch/infinigen/training_data_mix_all_balance/"
    "no_thinking/no_thinking.jsonl"
)
DEFAULT_BUNDLE_ROOT = "/path/to/scratch/infinigen/share_dataset"

PM_ANNO_ROOTS = {
    "v4":    "/path/to/scratch/anno_full/v4",
    "ankur": "/path/to/scratch/anno_full/ankur",
    "v5":    "/path/to/scratch/VisualCoT/training_data/map_questions_synth/annotation_v5",
}
DEFAULT_TOPDOWN_ROOT = "/path/to/scratch/infinigen/map_questions_topdown_blender"
DEFAULT_CORNER_VIEW_ROOT = "/path/to/scratch/infinigen/map_questions_corner_view"
DEFAULT_CORNER_VIEW_FILENAME = "corner_view.png"

# Panorama variant (mix_all visual_only) — same source files as create_mix_all_sft_data.py
EXISTING_TRAIN_FILE = "/path/to/scratch/VisualCoT/training_data/panorama_train_samples_filtered.json"
EXISTING_MAPPING_FILE = "/path/to/scratch/VisualCoT/training_data/scene_to_sample_id_filtered.json"
EXISTING_PANORAMA_DIR = "/path/to/scratch/VisualCoT/infinigen/rendered_panorama_train"
V4_TRAIN_FILES = [
    "/path/to/scratch/infinigen/outputs_rendered/dataset_counting_questions_filtered_V4.json",
    "/path/to/scratch/infinigen/dataset_anchor_questions_filtered_V4_normalized.json",
    "/path/to/scratch/infinigen/dataset_spatial_questions_filtered_V4_normalized.json",
    "/path/to/scratch/infinigen/outputs_rendered/dataset_relative_distance_questions_filtered_V4.json",
    "/path/to/scratch/infinigen/dataset_perspective_taking_questions_filtered_V4_normalized.json",
]
V4_MAPPING_FILE = "/path/to/scratch/infinigen/scene_to_sample_id_v4_filtered.json"
V4_PANORAMA_DIR = "/path/to/scratch/infinigen/rendered_panorama_v4"
V5_TRAIN_FILES = [
    "/path/to/scratch/infinigen/spatial/dataset_counting_questions_filtered_V5_normalized.json",
    "/path/to/scratch/infinigen/spatial/dataset_anchor_questions_filtered_V5_normalized.json",
    "/path/to/scratch/infinigen/spatial/dataset_spatial_questions_filtered_V5_normalized.json",
    "/path/to/scratch/infinigen/spatial/dataset_relative_distance_questions_filtered_V5.json",
    "/path/to/scratch/infinigen/spatial/dataset_perspective_taking_questions_filtered_V5_normalized.json",
]
V5_MAPPING_FILE = "/path/to/scratch/infinigen/scene_to_sample_id_v5_filtered.json"
V5_PANORAMA_DIR = "/path/to/scratch/infinigen/rendered_panorama_v5"
BAD_PANORAMAS_FILE = "/path/to/scratch/infinigen/bad_panoramas.json"

SOURCE_PREFIXES = ("v4_", "v5_", "ankur_")


# ─── Helpers ported from the source scripts ─────────────────────────────────────

def remap_path(path):
    if path is None:
        return None
    path = path.replace("/path/to/scratch", "/path/to/scratch")
    path = path.replace("/path/to/scratch", "/path/to/scratch")
    return path


def detect_source_from_path(image_path):
    """For paths under /network/scratch/{q,a}/.../ — returns v4/v5/ankur."""
    if not image_path:
        return None
    if "/infinigen/outputs_rendered/" in image_path:
        return "v4"
    if "/infinigen/spatial/" in image_path:
        return "v5"
    if "spatial_collab_dataset" in image_path or "v00_filtered" in image_path:
        return "ankur"
    return None


def extract_scene_id(image_path):
    parts = image_path.split("/")
    try:
        i = parts.index("frames")
        return parts[i - 1]
    except ValueError:
        return None


def infer_source_for_panorama_sample(image_path):
    """Mirrors create_mix_all_sft_data.infer_source_from_path."""
    if not image_path:
        return "unknown"
    if "infinigen/spatial/" in image_path:
        return "v5"
    if "outputs_rendered/" in image_path:
        return "v4"
    if "spatial_collab_dataset/scenes/" in image_path:
        return "ankur"
    return "unknown"


def prefixed_sample_id(sample):
    sid = sample.get("sample_id", "")
    if not sid:
        return sid
    if sid.startswith(SOURCE_PREFIXES):
        return sid
    src = infer_source_for_panorama_sample(sample.get("user_1_image_local_path", ""))
    return f"{src}_{sid}" if src != "unknown" else sid


def resolve_panorama_path(sample, existing_mapping, v4_mapping, v5_mapping,
                          existing_panorama_dir, v4_panorama_dir, v5_panorama_dir):
    scene_id = sample.get("scene_id")
    if scene_id in existing_mapping:
        first = existing_mapping[scene_id]
        return os.path.join(existing_panorama_dir, scene_id, f"panorama_blender_limits_{first}.png")
    if scene_id in v4_mapping:
        first = v4_mapping[scene_id]
        return os.path.join(v4_panorama_dir, scene_id, f"panorama_blender_limits_{first}.png")
    if scene_id in v5_mapping:
        first = v5_mapping[scene_id]
        return os.path.join(v5_panorama_dir, scene_id, f"panorama_blender_limits_{first}.png")
    return None


def get_question_and_answer(sample):
    question = sample.get("question_both_views", "").strip()
    options_user_1 = sample.get("options_user_1")
    options_user_2 = sample.get("options_user_2")
    if options_user_1 is not None:
        options = options_user_1
        correct_answer_idx = sample.get("user_1_gt_answer_idx")
    elif options_user_2 is not None:
        options = options_user_2
        correct_answer_idx = sample.get("user_2_gt_answer_idx")
    else:
        options = sample.get("options", [])
        correct_answer_idx = sample.get("correct_answer_idx")
    if correct_answer_idx is not None:
        answer_text = chr(65 + correct_answer_idx)
    else:
        answer_text = sample.get("correct_answer", "").strip()
    return question, options, answer_text


def extract_question_from_human(human_value):
    """The no_thinking human prompt ends with `\n<image><image>\n<question>...`."""
    last = human_value.rfind("<image>")
    tail = human_value[last + len("<image>"):] if last != -1 else human_value
    return tail.lstrip("\n").strip()


def extract_answer_letter(gpt_value):
    """gpt is '<answer>X</answer>' — return X (or the whole content)."""
    s = (gpt_value or "").strip()
    if s.startswith("<answer>") and s.endswith("</answer>"):
        return s[len("<answer>"): -len("</answer>")].strip()
    return s


# ─── Image copy helpers ─────────────────────────────────────────────────────────

def copy_png(src_abs, dst_abs):
    """Copy src_abs to dst_abs if dst doesn't exist. Idempotent."""
    if os.path.exists(dst_abs):
        return True
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    try:
        shutil.copy2(src_abs, dst_abs)
        return True
    except Exception as e:
        print(f"[ERROR] copy {src_abs} -> {dst_abs}: {e}")
        return False


def build_pm_bridge(annotated_cam0, annotated_cam1, dst_abs):
    """Side-by-side composite. Skip if dst exists."""
    if os.path.exists(dst_abs):
        return True
    try:
        with Image.open(annotated_cam0) as i0, Image.open(annotated_cam1) as i1:
            if i0.mode != "RGB":
                i0 = i0.convert("RGB")
            if i1.mode != "RGB":
                i1 = i1.convert("RGB")
            h = max(i0.height, i1.height)
            w = i0.width + i1.width
            canvas = Image.new("RGB", (w, h), (0, 0, 0))
            canvas.paste(i0, (0, 0))
            canvas.paste(i1, (i0.width, 0))
            os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
            canvas.save(dst_abs, format="PNG")
        return True
    except Exception as e:
        print(f"[ERROR] pm_bridge {dst_abs}: {e}")
        return False


# ─── Pass 1: parsing & id-set collection ───────────────────────────────────────

def parse_no_thinking_row(row, image_prefix):
    """Returns dict with sample_id, src, scene_id, cam0_abs, cam1_abs, plus the
    original conversations/image fields — or None if unparsable / not eligible."""
    sample_id = row.get("id")
    if not sample_id or "rotation" in sample_id:
        return None
    images = row.get("image") or []
    if len(images) < 2:
        return None
    img0_rel = remap_path(images[0])
    img1_rel = remap_path(images[1])
    img0_abs = img0_rel if img0_rel.startswith("/") else os.path.join(image_prefix, img0_rel)
    img1_abs = img1_rel if img1_rel.startswith("/") else os.path.join(image_prefix, img1_rel)
    src = detect_source_from_path(img0_abs)
    scene = extract_scene_id(img0_abs)
    if not src or not scene:
        return None
    return {
        "id":         sample_id,
        "src":        src,
        "scene":      scene,
        "cam0_abs":   img0_abs,
        "cam1_abs":   img1_abs,
        "row":        row,
    }


def collect_no_thinking_ids(jsonl_path, image_prefix):
    """Returns {id: parsed_dict} for every row whose cam0/cam1 exist on disk."""
    out = {}
    status = Counter()
    with open(jsonl_path) as f:
        for ln in f:
            row = json.loads(ln)
            parsed = parse_no_thinking_row(row, image_prefix)
            if parsed is None:
                status["unparsable"] += 1
                continue
            if not (os.path.exists(parsed["cam0_abs"]) and os.path.exists(parsed["cam1_abs"])):
                status["no_cam_images"] += 1
                continue
            out[parsed["id"]] = parsed
            status["ok"] += 1
    return out, status


def collect_pm_ids(no_thinking_parsed, pm_roots):
    """Filter no_thinking ids to those that ALSO have PM annotations."""
    out = {}
    status = Counter()
    for sid, p in no_thinking_parsed.items():
        anno_root = pm_roots.get(p["src"])
        if anno_root is None:
            status["unknown_source"] += 1
            continue
        a0 = os.path.join(anno_root, p["scene"], "annotated_cam0.png")
        a1 = os.path.join(anno_root, p["scene"], "annotated_cam1.png")
        if not (os.path.exists(a0) and os.path.exists(a1)):
            status["no_pm_annotation"] += 1
            continue
        out[sid] = {**p, "annotated_cam0": a0, "annotated_cam1": a1}
        status["ok"] += 1
    return out, status


def collect_topdown_ids(no_thinking_parsed, topdown_root):
    out = {}
    status = Counter()
    for sid, p in no_thinking_parsed.items():
        topdown_abs = os.path.join(topdown_root, p["scene"], "topdown_blender.png")
        if not os.path.exists(topdown_abs):
            status["no_topdown"] += 1
            continue
        out[sid] = {**p, "topdown_abs": topdown_abs}
        status["ok"] += 1
    return out, status


def collect_corner_view_ids(no_thinking_parsed, corner_view_root, filename):
    out = {}
    status = Counter()
    for sid, p in no_thinking_parsed.items():
        cv_abs = os.path.join(corner_view_root, p["scene"], filename)
        if not os.path.exists(cv_abs):
            status["no_corner_view"] += 1
            continue
        out[sid] = {**p, "corner_view_abs": cv_abs}
        status["ok"] += 1
    return out, status


def load_panorama_samples(args, seed=42):
    """Replicate the create_mix_all_sft_data.py pipeline through `all_samples`.

    Returns a list of sample dicts, each with sample_id (prefixed), scene_id,
    user_1_image_local_path, user_2_image_local_path, panorama_path (resolved).
    """
    random.seed(seed)

    def load(filepath):
        if not os.path.exists(filepath):
            print(f"  [SKIP] missing: {filepath}")
            return []
        with open(filepath) as f:
            return json.load(f)

    print(f"\n[panorama] Loading existing/V4/V5 samples...")
    existing = load(args.existing_train_file)
    v4 = []
    for fp in args.v4_train_files:
        v4.extend(load(fp))
    v5 = []
    for fp in args.v5_train_files:
        v5.extend(load(fp))
    print(f"  existing={len(existing)}  v4={len(v4)}  v5={len(v5)}")

    all_samples = existing + v4 + v5

    # mappings
    existing_mapping = json.load(open(args.existing_mapping_file)) if os.path.exists(args.existing_mapping_file) else {}
    v4_mapping       = json.load(open(args.v4_mapping_file))       if os.path.exists(args.v4_mapping_file)       else {}
    v5_mapping       = json.load(open(args.v5_mapping_file))       if os.path.exists(args.v5_mapping_file)       else {}

    # filter: must have a non-empty question_both_views
    all_samples = [s for s in all_samples if s.get("question_both_views", "").strip()]

    # bad panorama blacklist
    bad = set()
    if os.path.exists(BAD_PANORAMAS_FILE):
        with open(BAD_PANORAMAS_FILE) as f:
            bad = set(json.load(f).get("bad_panoramas", []))
        def has_bad(s):
            p = resolve_panorama_path(s, existing_mapping, v4_mapping, v5_mapping,
                                      args.existing_panorama_dir, args.v4_panorama_dir, args.v5_panorama_dir)
            return p in bad
        before = len(all_samples)
        all_samples = [s for s in all_samples if not has_bad(s)]
        print(f"  filtered {before - len(all_samples)} bad-panorama samples")

    # prefix sample_ids
    for s in all_samples:
        new_sid = prefixed_sample_id(s)
        if new_sid != s.get("sample_id", ""):
            s["sample_id"] = new_sid

    # max_relative_distance balanced cap (anchor vs camera, 50/50)
    if args.max_relative_distance > 0:
        anchor_types = {"closest", "farthest", "relative_distance"}
        camera_types = {"closest_to_camera_1", "closest_to_camera_2",
                        "farthest_from_camera_1", "farthest_from_camera_2"}
        anchor_qs = [s for s in all_samples if s.get("question_type") in anchor_types]
        camera_qs = [s for s in all_samples if s.get("question_type") in camera_types]
        other     = [s for s in all_samples if s.get("question_type") not in anchor_types | camera_types]
        n_each = args.max_relative_distance // 2
        if len(anchor_qs) > n_each:
            anchor_qs = random.sample(anchor_qs, n_each)
        if len(camera_qs) > n_each:
            camera_qs = random.sample(camera_qs, n_each)
        all_samples = other + anchor_qs + camera_qs

    # max_perspective_taking + spatial_orientation cap (50/50)
    if args.max_perspective_taking > 0:
        persp = [s for s in all_samples if s.get("question_type") == "perspective_taking"]
        spat  = [s for s in all_samples if s.get("question_type") == "spatial_orientation"]
        other = [s for s in all_samples if s.get("question_type") not in ("perspective_taking", "spatial_orientation")]
        n_each = args.max_perspective_taking // 2
        if len(persp) > n_each:
            persp = random.sample(persp, n_each)
        if len(spat) > n_each:
            spat = random.sample(spat, n_each)
        all_samples = other + persp + spat

    # Resolve panorama + cam paths for each sample
    enriched = []
    for s in all_samples:
        cam0 = remap_path(s.get("user_1_image_local_path"))
        cam1 = remap_path(s.get("user_2_image_local_path"))
        pano = resolve_panorama_path(s, existing_mapping, v4_mapping, v5_mapping,
                                     args.existing_panorama_dir, args.v4_panorama_dir, args.v5_panorama_dir)
        if not all([cam0, cam1, pano]):
            continue
        src = detect_source_from_path(cam0)
        scene = extract_scene_id(cam0)
        if not src or not scene:
            continue
        enriched.append({
            "sample": s,
            "id":        s.get("sample_id", ""),
            "src":       src,
            "scene":     scene,
            "cam0_abs":  cam0,
            "cam1_abs":  cam1,
            "panorama_abs": pano,
        })
    print(f"  enriched: {len(enriched)} samples with resolved paths")
    return enriched


def collect_panorama_ids(args):
    """Legacy: use the create_mix_all_sft_data.py pipeline to resolve panorama
    paths. This misses ankur samples because their source JSON was deleted."""
    enriched = load_panorama_samples(args)
    out = {}
    status = Counter()
    for p in enriched:
        if not p["id"]:
            status["no_id"] += 1
            continue
        if p["id"] in out:
            status["dup_id"] += 1
            continue
        if not (os.path.exists(p["cam0_abs"]) and os.path.exists(p["cam1_abs"])
                and os.path.exists(p["panorama_abs"])):
            status["missing_files"] += 1
            continue
        out[p["id"]] = p
        status["ok"] += 1
    return out, status


def collect_panorama_from_parquet(nt_parsed, parquet_dir, bundle_root, max_workers=16):
    """Recover panorama bytes from the mix_all visual_only parquet chunks.

    For each no_thinking sample, the on-disk cam0 PNG bytes are matched
    against the parquet's embedded `image_list[0]` to find the corresponding
    parquet row, whose `image_list[2]` is the panorama PNG bytes.

    Returns ({id: parsed_dict_with_panorama_bytes}, status_counter).
    parsed_dict_with_panorama_bytes has:
      - all the no_thinking parse fields (id, src, scene, cam0_abs, cam1_abs, row)
      - panorama_png_bytes: bytes
    """
    import hashlib
    import pyarrow.parquet as pq
    import glob

    # Step 1: hash every unique on-disk cam0 PNG (one per scene) and build
    # {cam0_hash: scene_info}. Different ids/questions can share the same cam0,
    # so we map hash → list of no_thinking entries; once we find the matching
    # parquet row's panorama, all ids in that group get it.
    print("  [pano-parquet] hashing on-disk cam0 PNGs ...")
    unique_cam0 = {}  # cam0_abs -> first parsed dict using it
    cam0_to_ids = {}  # cam0_abs -> [ids]
    for sid, p in nt_parsed.items():
        unique_cam0.setdefault(p["cam0_abs"], p)
        cam0_to_ids.setdefault(p["cam0_abs"], []).append(sid)
    print(f"    {len(unique_cam0)} unique cam0 paths")

    cam0_hash_to_path = {}

    def hash_file(path):
        try:
            with open(path, "rb") as f:
                return path, hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return path, None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for path, h in tqdm(ex.map(hash_file, unique_cam0.keys()),
                            total=len(unique_cam0), desc="hashing cam0"):
            if h is not None:
                cam0_hash_to_path[h] = path

    print(f"    hashed {len(cam0_hash_to_path)} cam0 PNGs successfully")

    # Step 2: iterate every parquet row; if its cam0 hash matches one of our
    # on-disk cam0s, store the panorama bytes under that scene.
    chunks = sorted(glob.glob(os.path.join(parquet_dir, "chunk_*.parquet")))
    print(f"  [pano-parquet] scanning {len(chunks)} parquet chunks for panorama bytes")

    scene_to_panorama_bytes = {}  # scene_abs_path (cam0_abs) -> panorama_bytes
    pq_total_rows = 0
    pq_matched = 0
    pq_unmatched = 0
    for chunk in tqdm(chunks, desc="parquet scan"):
        t = pq.read_table(chunk, columns=["image_list"])
        n = t.num_rows
        pq_total_rows += n
        # Pull each row's image_list (list of binary)
        rows = t.to_pylist()
        for r in rows:
            imgs = r["image_list"]
            if not imgs or len(imgs) < 3:
                continue
            cam0_h = hashlib.sha256(imgs[0]).hexdigest()
            path = cam0_hash_to_path.get(cam0_h)
            if path is None:
                pq_unmatched += 1
                continue
            # Multiple parquet rows can share a cam0 (different question/scene).
            # Panorama is per-scene → same across rows. First wins.
            if path not in scene_to_panorama_bytes:
                scene_to_panorama_bytes[path] = imgs[2]
            pq_matched += 1

    print(f"    parquet rows: {pq_total_rows} total, {pq_matched} matched a no_thinking cam0, "
          f"{pq_unmatched} unmatched")
    print(f"    recovered panorama bytes for {len(scene_to_panorama_bytes)}/{len(unique_cam0)} scenes")

    # Step 3: build the panorama parsed dict for every no_thinking id whose
    # scene has recovered panorama bytes.
    out = {}
    status = Counter()
    for sid, p in nt_parsed.items():
        bytes_ = scene_to_panorama_bytes.get(p["cam0_abs"])
        if bytes_ is None:
            status["no_panorama_in_parquet"] += 1
            continue
        out[sid] = {**p, "panorama_bytes": bytes_}
        status["ok"] += 1
    return out, status


# ─── Pass 2: image copy + JSONL emit ────────────────────────────────────────────

def rel_image_path(src, scene, name):
    return f"images/{src}/{scene}/{name}"


def emit_no_thinking(parsed, bundle_root, image_paths_rel):
    """Build the JSONL row for no_thinking — preserve original conversations
    but rewrite image[] paths to the new bundle-relative form."""
    row = parsed["row"]
    return {
        "conversations": row["conversations"],
        "image":         image_paths_rel,
        "id":            parsed["id"],
    }


def build_visual_only_row(parsed, bundle_root, cam_paths_rel, visual_thinking_rel,
                          no_thinking_row):
    """Build a visual_only row from the no_thinking source row.

    Inputs are still the 2 camera views (<image><image> placeholders); the
    bridge image is the *output* of visual thinking (between <image_start>
    and <image_end>), so it lives in a separate `visual_thinking` field, not
    in `image`.

    - human: swap NO_THINKING for VISUAL_ONLY_THINK system prompt; keep <image><image>.
    - gpt:   "<image_start><image_end>" + original gpt ("<answer>X</answer>").
    """
    src_human = no_thinking_row["conversations"][0]["value"]
    src_gpt   = no_thinking_row["conversations"][1]["value"]
    new_human = src_human.replace(NO_THINKING_SYSTEM_PROMPT, VISUAL_ONLY_THINK_SYSTEM_PROMPT, 1)
    new_gpt   = "<image_start><image_end>" + src_gpt
    return {
        "conversations": [
            {"from": "human", "value": new_human},
            {"from": "gpt",   "value": new_gpt},
        ],
        "image":           cam_paths_rel,
        "visual_thinking": visual_thinking_rel,
        "id":              parsed["id"],
    }


def write_panorama_bytes(panorama_bytes, dst_abs):
    """Write panorama PNG bytes to disk. Idempotent."""
    if os.path.exists(dst_abs):
        return True
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    try:
        with open(dst_abs, "wb") as f:
            f.write(panorama_bytes)
        return True
    except Exception as e:
        print(f"[ERROR] write panorama {dst_abs}: {e}")
        return False


def write_pass2(bundle_root, keep_ids, nt_parsed, pm_parsed, td_parsed, pn_parsed,
                cv_parsed, no_thinking_jsonl_source, max_workers=16):
    """Pass 2: copy PNGs + write 4 JSONLs."""
    os.makedirs(bundle_root, exist_ok=True)

    # Build {id: source_row} from no_thinking.jsonl so visual variants can clone fields
    src_row_by_id = {}
    with open(no_thinking_jsonl_source) as f:
        for ln in f:
            r = json.loads(ln)
            if r.get("id") in keep_ids:
                src_row_by_id[r["id"]] = r

    # Sanity: panorama variant rows aren't in no_thinking.jsonl by id matching?
    # They should be, since both come from create_mix_all_sft_data with same filters.
    # But the panorama variant builds its own conversation from the sample dict
    # (because its question/answer source is the V4/V5 JSONs, not no_thinking).
    # We'll use src_row_by_id for pm/topdown, and the sample dict for panorama.

    # ── no_thinking ──
    nt_path = os.path.join(bundle_root, "no_thinking.jsonl")
    print(f"\n[pass2] Writing no_thinking → {nt_path}")
    written = 0
    with open(nt_path, "w") as out:
        for sid in tqdm(keep_ids):
            p = nt_parsed[sid]
            rel0 = rel_image_path(p["src"], p["scene"], "cam0.png")
            rel1 = rel_image_path(p["src"], p["scene"], "cam1.png")
            ok0 = copy_png(p["cam0_abs"], os.path.join(bundle_root, rel0))
            ok1 = copy_png(p["cam1_abs"], os.path.join(bundle_root, rel1))
            if not (ok0 and ok1):
                continue
            row = emit_no_thinking(p, bundle_root, [rel0, rel1])
            out.write(json.dumps(row) + "\n")
            written += 1
    print(f"  {written} rows written")

    # ── PM ──
    pm_path = os.path.join(bundle_root, "visual_only_pm.jsonl")
    print(f"\n[pass2] Writing PM → {pm_path}")
    written = 0
    with open(pm_path, "w") as out:
        for sid in tqdm(keep_ids):
            p = pm_parsed[sid]
            rel0 = rel_image_path(p["src"], p["scene"], "cam0.png")
            rel1 = rel_image_path(p["src"], p["scene"], "cam1.png")
            relB = rel_image_path(p["src"], p["scene"], "pm_bridge.png")
            ok0 = copy_png(p["cam0_abs"], os.path.join(bundle_root, rel0))
            ok1 = copy_png(p["cam1_abs"], os.path.join(bundle_root, rel1))
            okB = build_pm_bridge(p["annotated_cam0"], p["annotated_cam1"],
                                  os.path.join(bundle_root, relB))
            if not (ok0 and ok1 and okB):
                continue
            row = build_visual_only_row(p, bundle_root, [rel0, rel1], relB,
                                        src_row_by_id[sid])
            out.write(json.dumps(row) + "\n")
            written += 1
    print(f"  {written} rows written")

    # ── topdown ──
    td_path = os.path.join(bundle_root, "visual_only_topdown.jsonl")
    print(f"\n[pass2] Writing topdown → {td_path}")
    written = 0
    with open(td_path, "w") as out:
        for sid in tqdm(keep_ids):
            p = td_parsed[sid]
            rel0 = rel_image_path(p["src"], p["scene"], "cam0.png")
            rel1 = rel_image_path(p["src"], p["scene"], "cam1.png")
            relT = rel_image_path(p["src"], p["scene"], "topdown_blender.png")
            ok0 = copy_png(p["cam0_abs"], os.path.join(bundle_root, rel0))
            ok1 = copy_png(p["cam1_abs"], os.path.join(bundle_root, rel1))
            okT = copy_png(p["topdown_abs"], os.path.join(bundle_root, relT))
            if not (ok0 and ok1 and okT):
                continue
            row = build_visual_only_row(p, bundle_root, [rel0, rel1], relT,
                                        src_row_by_id[sid])
            out.write(json.dumps(row) + "\n")
            written += 1
    print(f"  {written} rows written")

    # ── panorama ──
    # Panorama bytes come from the parquet (not a path on disk), so we write
    # them out directly. The row's question/answer come from no_thinking.jsonl
    # (id-aligned), so build_visual_only_row works the same as PM/topdown.
    pn_path = os.path.join(bundle_root, "visual_only_panorama.jsonl")
    print(f"\n[pass2] Writing panorama → {pn_path}")
    written = 0
    with open(pn_path, "w") as out:
        for sid in tqdm(keep_ids):
            p = pn_parsed[sid]
            rel0 = rel_image_path(p["src"], p["scene"], "cam0.png")
            rel1 = rel_image_path(p["src"], p["scene"], "cam1.png")
            relP = rel_image_path(p["src"], p["scene"], "panorama.png")
            ok0 = copy_png(p["cam0_abs"], os.path.join(bundle_root, rel0))
            ok1 = copy_png(p["cam1_abs"], os.path.join(bundle_root, rel1))
            okP = write_panorama_bytes(p["panorama_bytes"],
                                       os.path.join(bundle_root, relP))
            if not (ok0 and ok1 and okP):
                continue
            row = build_visual_only_row(p, bundle_root, [rel0, rel1], relP,
                                        src_row_by_id[sid])
            out.write(json.dumps(row) + "\n")
            written += 1
    print(f"  {written} rows written")

    # ── corner_view ──
    cv_path = os.path.join(bundle_root, "visual_only_corner_view.jsonl")
    print(f"\n[pass2] Writing corner_view → {cv_path}")
    written = 0
    with open(cv_path, "w") as out:
        for sid in tqdm(keep_ids):
            p = cv_parsed[sid]
            rel0 = rel_image_path(p["src"], p["scene"], "cam0.png")
            rel1 = rel_image_path(p["src"], p["scene"], "cam1.png")
            relC = rel_image_path(p["src"], p["scene"], "corner_view.png")
            ok0 = copy_png(p["cam0_abs"], os.path.join(bundle_root, rel0))
            ok1 = copy_png(p["cam1_abs"], os.path.join(bundle_root, rel1))
            okC = copy_png(p["corner_view_abs"], os.path.join(bundle_root, relC))
            if not (ok0 and ok1 and okC):
                continue
            row = build_visual_only_row(p, bundle_root, [rel0, rel1], relC,
                                        src_row_by_id[sid])
            out.write(json.dumps(row) + "\n")
            written += 1
    print(f"  {written} rows written")


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no_thinking_jsonl", default=DEFAULT_NO_THINKING_JSONL)
    p.add_argument("--bundle_root", default=DEFAULT_BUNDLE_ROOT)
    p.add_argument("--image_prefix", default="/network/scratch",
                   help="Prefix for relative image paths in no_thinking.jsonl")
    p.add_argument("--pm_v4_root",    default=PM_ANNO_ROOTS["v4"])
    p.add_argument("--pm_ankur_root", default=PM_ANNO_ROOTS["ankur"])
    p.add_argument("--pm_v5_root",    default=PM_ANNO_ROOTS["v5"])
    p.add_argument("--topdown_root",  default=DEFAULT_TOPDOWN_ROOT)
    p.add_argument("--corner_view_root", default=DEFAULT_CORNER_VIEW_ROOT)
    p.add_argument("--corner_view_filename", default=DEFAULT_CORNER_VIEW_FILENAME)
    # Panorama-variant inputs
    p.add_argument("--existing_train_file",    default=EXISTING_TRAIN_FILE)
    p.add_argument("--existing_mapping_file",  default=EXISTING_MAPPING_FILE)
    p.add_argument("--existing_panorama_dir",  default=EXISTING_PANORAMA_DIR)
    p.add_argument("--v4_train_files", nargs="+", default=V4_TRAIN_FILES)
    p.add_argument("--v4_mapping_file",        default=V4_MAPPING_FILE)
    p.add_argument("--v4_panorama_dir",        default=V4_PANORAMA_DIR)
    p.add_argument("--v5_train_files", nargs="+", default=V5_TRAIN_FILES)
    p.add_argument("--v5_mapping_file",        default=V5_MAPPING_FILE)
    p.add_argument("--v5_panorama_dir",        default=V5_PANORAMA_DIR)
    p.add_argument("--max_relative_distance",  type=int, default=3000)
    p.add_argument("--max_perspective_taking", type=int, default=3000)
    p.add_argument("--mix_all_visual_only_parquet_dir",
                   default="/path/to/scratch/infinigen/training_data_mix_all_balance/visual_only",
                   help="Source dir for panorama-bytes recovery (default: the actual training parquets).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_workers", type=int, default=16)
    p.add_argument("--dry_run", action="store_true",
                   help="Pass 1 only — print intersection size, no writes.")
    args = p.parse_args()

    pm_roots = {"v4": args.pm_v4_root, "ankur": args.pm_ankur_root, "v5": args.pm_v5_root}

    # Pass 1
    print(f"\n[pass1] no_thinking ids from {args.no_thinking_jsonl}")
    nt_parsed, nt_status = collect_no_thinking_ids(args.no_thinking_jsonl, args.image_prefix)
    print(f"  status: {dict(nt_status)}")

    print(f"\n[pass1] PM ids (anno roots: {pm_roots})")
    pm_parsed, pm_status = collect_pm_ids(nt_parsed, pm_roots)
    print(f"  status: {dict(pm_status)}")

    print(f"\n[pass1] topdown ids (root: {args.topdown_root})")
    td_parsed, td_status = collect_topdown_ids(nt_parsed, args.topdown_root)
    print(f"  status: {dict(td_status)}")

    print(f"\n[pass1] corner_view ids (root: {args.corner_view_root}, file: {args.corner_view_filename})")
    cv_parsed, cv_status = collect_corner_view_ids(nt_parsed, args.corner_view_root, args.corner_view_filename)
    print(f"  status: {dict(cv_status)}")

    print(f"\n[pass1] panorama ids (from {args.mix_all_visual_only_parquet_dir})")
    pn_parsed, pn_status = collect_panorama_from_parquet(
        nt_parsed, args.mix_all_visual_only_parquet_dir, args.bundle_root,
        max_workers=args.max_workers,
    )
    print(f"  status: {dict(pn_status)}")

    nt_ids = set(nt_parsed.keys())
    pm_ids = set(pm_parsed.keys())
    td_ids = set(td_parsed.keys())
    pn_ids = set(pn_parsed.keys())
    cv_ids = set(cv_parsed.keys())

    print("\n" + "=" * 60)
    print(f"  no_thinking : {len(nt_ids)} ids available")
    print(f"  pm          : {len(pm_ids)} ids available")
    print(f"  topdown     : {len(td_ids)} ids available")
    print(f"  panorama    : {len(pn_ids)} ids available")
    print(f"  corner_view : {len(cv_ids)} ids available")

    keep = nt_ids & pm_ids & td_ids & pn_ids & cv_ids
    print(f"  intersection: {len(keep)} ids  (kept in all 5 JSONLs)")
    print("\nPer-variant losses vs no_thinking baseline:")
    print(f"  pm  miss          : {len(nt_ids - pm_ids)}")
    print(f"  topdown miss      : {len(nt_ids - td_ids)}")
    print(f"  panorama miss     : {len(nt_ids - pn_ids)}")
    print(f"  corner_view miss  : {len(nt_ids - cv_ids)}")
    print("=" * 60)

    if args.dry_run:
        print("\n[dry_run] No writes. Exiting.")
        return

    write_pass2(args.bundle_root, sorted(keep),
                nt_parsed, pm_parsed, td_parsed, pn_parsed, cv_parsed,
                args.no_thinking_jsonl, max_workers=args.max_workers)

    print(f"\n[done] Bundle at {args.bundle_root}")
    print(f"  du -sh {args.bundle_root}  # to check size")


if __name__ == "__main__":
    main()
