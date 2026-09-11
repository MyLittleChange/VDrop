#!/usr/bin/env python3
"""
Gemini-generated oracle views (T_td_gemini / T_pano_gemini / T_cor_gemini)
for the Two-Reader Informativeness experiment.

Prompts gemini-3-pro-image-preview with V1 + V2 + a task-specific
instruction and saves the returned image(s) as PNGs.

Output layout:
    <root>/topdown/<scene_id>/topdown_gemini_<sample_id>.png
    <root>/panorama/<scene_id>/panorama_gemini_<sample_id>.png
    <root>/cor/<scene_id>/cor_gemini_<sample_id>_v{1,2}.png

Pilot subset: by default reads the same 100 samples used by the audit
HTML (deterministic selection based on oracle-coverage). Override with
--sample_ids or --all to expand.
"""
import argparse
import base64
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO


PROMPTS = {
    "topdown": (
        "You are given two camera images of the same indoor scene. "
        "Generate a single bird's-eye-view (top-down) image of the room. "
        "Show all major furniture and structural elements as if seen from "
        "directly above, with the wall layout. Match the spatial layout of "
        "the two input views. A clean photoreal-style top-down is preferred; "
        "an axonometric schematic is acceptable. Do not include any text "
        "labels, legends, or arrows."
    ),
    "panorama": (
        "You are given two camera images of the same indoor scene. "
        "Generate a single wide panoramic view (roughly 360 degrees, or a "
        "wide-angle 180-degree view) from a camera position that spans the "
        "union of the two input views' fields of view. Keep object "
        "identities and positions consistent with both inputs. Do not "
        "include any text labels."
    ),
    # T_cor is generated in two sequential calls (see generate_cor_pair).
    # Call 1: annotate IMAGE 1 (V1) with up to 3 numbered dots on co-visible objects.
    # Call 2: pass V1, V2, *and the annotated V1* back in and ask Gemini
    #         to apply the same colour/number scheme to V2's matching objects.
    "cor_v1": (
        "You are given two camera images of the same indoor scene "
        "(IMAGE 1 and IMAGE 2). On IMAGE 1 only, overlay up to three "
        "small filled coloured circles, each labelled with a white digit "
        "(1, 2, 3), placed on objects that are clearly visible in BOTH "
        "IMAGE 1 and IMAGE 2. Use distinct colours: dot 1 = red, dot 2 = "
        "green, dot 3 = blue. Pick prominent, easy-to-recognise objects "
        "(e.g. large furniture, lamps, windows). Return only the "
        "annotated version of IMAGE 1. Do not return IMAGE 2. Do not "
        "include any other text or annotation."
    ),
    "cor_v2": (
        "You are given two camera images of the same indoor scene. The "
        "first input is the ORIGINAL IMAGE 2. The second input is the "
        "ANNOTATED IMAGE 1, which already has up to three coloured dots "
        "(1 = red, 2 = green, 3 = blue) placed on objects visible in "
        "both views. Your task: take the ORIGINAL IMAGE 2 and overlay "
        "dots with the SAME colour and SAME number on the SAME "
        "real-world objects. For example, if the red '1' dot in the "
        "annotated reference is on a sofa, place a red '1' dot on that "
        "same sofa in IMAGE 2. Return only the annotated version of "
        "IMAGE 2. Do not return any other image. Do not include text "
        "or annotation beyond the three numbered dots."
    ),
}


def load_sample_ids_from_audit():
    """Reuse the audit HTML's selection logic: top 100 by oracle coverage,
    seeded random shuffle within each subtask. Returns a list of
    (subtask, sample_id) tuples."""
    idx_path = Path(
        "/path/to/scratch/VisualCoT/infinigen/"
        "two_reader_informativeness/two_reader_informativeness_index.json"
    )
    idx = json.load(open(idx_path))
    by_subtask = {}
    for r in idx["samples"]:
        by_subtask.setdefault(r["subtask"], []).append(r)
    # Match audit's selection: prefer rows with the most existing oracle views
    rng = random.Random(42)
    per_st = {}
    PER = 25
    for st, rows in by_subtask.items():
        rows = sorted(
            rows,
            key=lambda r: -sum(bool(r.get(k)) for k in
                               ("T_td", "T_td_blender", "T_pano", "T_cor", "T_cot", "T_noise"))
        )
        top = rows[: PER * 3]
        rng.shuffle(top)
        per_st[st] = top[:PER]
    # Interleave so --max_samples 4 gives 1 per subtask, 8 gives 2 per subtask, etc.
    out = []
    for i in range(PER):
        for st in ("anchor", "counting", "relative_distance", "relative_direction"):
            if i < len(per_st.get(st, [])):
                out.append(per_st[st][i])
    return out  # list of full provenance rows


def encode_png_bytes(path):
    """Read PNG bytes from disk."""
    with open(path, "rb") as f:
        return f.read()


def call_with_retry(client, model, contents, max_retries=3, base_delay=4):
    """Gemini call with exponential backoff on rate-limit / transient errors."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
            )
            return response
        except Exception as e:
            msg = str(e)
            if attempt == max_retries - 1:
                raise
            # Common transient errors: 429 rate-limit, 500 server, 503 unavailable
            wait = base_delay * (2 ** attempt)
            print(f"  retry {attempt+1}/{max_retries} after {wait}s: {msg[:120]}", flush=True)
            time.sleep(wait)


def _to_pil(obj):
    """Coerce a Gemini image payload (PIL Image, Gemini types.Image with
    image_bytes, or raw bytes) to a PIL Image."""
    if isinstance(obj, Image.Image):
        return obj
    # Gemini types.Image: has .image_bytes
    raw = getattr(obj, "image_bytes", None)
    if raw is None:
        raw = getattr(obj, "data", None)
    if raw is None and isinstance(obj, (bytes, bytearray)):
        raw = bytes(obj)
    if raw is None:
        return None
    try:
        return Image.open(BytesIO(raw))
    except Exception:
        return None


def extract_images(response):
    """Pull all generated images out of a Gemini response.
    Always returns a list of PIL Images.
    """
    images = []
    # Layout 1: response.parts
    parts = getattr(response, "parts", None)
    if parts:
        for p in parts:
            if getattr(p, "inline_data", None) is not None:
                # Try .as_image() first (returns Gemini types.Image), then
                # coerce to PIL.
                img = None
                if hasattr(p, "as_image"):
                    try:
                        img = _to_pil(p.as_image())
                    except Exception:
                        img = None
                if img is None:
                    img = _to_pil(p.inline_data.data)
                if img is not None:
                    images.append(img)
    # Layout 2: response.candidates[0].content.parts
    if not images:
        try:
            for c in response.candidates or []:
                for p in (c.content.parts or []):
                    if getattr(p, "inline_data", None) is not None:
                        img = _to_pil(p.inline_data.data)
                        if img is not None:
                            images.append(img)
        except Exception:
            pass
    return images


def generate_cor_pair(client, model, sample, out_dir, force=False):
    """T_cor: two sequential calls.
       Call 1: annotate V1 with numbered dots on co-visible objects.
       Call 2: pass V2 + annotated V1 back and ask for matching dots on V2.
    Returns status dict with both PNG paths.
    """
    sid = sample["sample_id"]
    scene = sample["scene_id"]
    scene_dir = Path(out_dir) / "cor" / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    out_v1 = scene_dir / f"cor_gemini_{sid}_v1.png"
    out_v2 = scene_dir / f"cor_gemini_{sid}_v2.png"
    if not force and out_v1.exists() and out_v2.exists():
        return {"sample_id": sid, "T": "cor", "status": "skipped_exists",
                "paths": [str(out_v1), str(out_v2)]}

    v1_path = sample.get("_user_1_image") or sample.get("user_1_image_local_path")
    v2_path = sample.get("_user_2_image") or sample.get("user_2_image_local_path")
    try:
        img1_bytes = encode_png_bytes(v1_path)
        img2_bytes = encode_png_bytes(v2_path)
    except FileNotFoundError as e:
        return {"sample_id": sid, "T": "cor", "status": "error",
                "error": f"input image not found: {e}"}

    img1_part = types.Part.from_bytes(data=img1_bytes, mime_type="image/png")
    img2_part = types.Part.from_bytes(data=img2_bytes, mime_type="image/png")

    # Call 1: annotate V1
    try:
        resp1 = call_with_retry(client, model, [PROMPTS["cor_v1"], img1_part, img2_part])
    except Exception as e:
        return {"sample_id": sid, "T": "cor", "status": "error",
                "error": f"call1: {e}"}
    images1 = extract_images(resp1)
    if not images1:
        text = ""
        try:
            text = resp1.text or ""
        except Exception:
            pass
        return {"sample_id": sid, "T": "cor", "status": "no_images_v1",
                "refusal_text": text[:300]}
    annotated_v1 = images1[0]
    annotated_v1.save(out_v1)

    # Encode annotated V1 to pass as image input for call 2
    ann_buf = BytesIO()
    annotated_v1.convert("RGB").save(ann_buf, format="PNG")
    ann_bytes = ann_buf.getvalue()
    ann_part = types.Part.from_bytes(data=ann_bytes, mime_type="image/png")

    # Call 2: pass V2 (original) and annotated V1 (reference) — note the
    # prompt expects "ORIGINAL IMAGE 2" first, then "ANNOTATED IMAGE 1".
    try:
        resp2 = call_with_retry(client, model, [PROMPTS["cor_v2"], img2_part, ann_part])
    except Exception as e:
        return {"sample_id": sid, "T": "cor", "status": "ok_v1_only",
                "paths": [str(out_v1)], "error_v2": f"call2: {e}"}
    images2 = extract_images(resp2)
    if not images2:
        text = ""
        try:
            text = resp2.text or ""
        except Exception:
            pass
        return {"sample_id": sid, "T": "cor", "status": "ok_v1_only",
                "paths": [str(out_v1)], "refusal_text_v2": text[:300]}
    images2[0].save(out_v2)
    return {"sample_id": sid, "T": "cor", "status": "ok",
            "paths": [str(out_v1), str(out_v2)],
            "n_returned": (len(images1), len(images2))}


def generate_one(client, model, sample, T_name, out_dir, force=False):
    """Generate a Gemini oracle of type T_name for one COSMIC sample.

    For T_cor, delegates to generate_cor_pair (two sequential calls).
    Returns: dict with status + output paths.
    """
    if T_name == "cor":
        return generate_cor_pair(client, model, sample, out_dir, force=force)

    sid = sample["sample_id"]
    scene = sample["scene_id"]
    scene_dir = Path(out_dir) / T_name / scene
    scene_dir.mkdir(parents=True, exist_ok=True)

    out = scene_dir / f"{T_name}_gemini_{sid}.png"
    outs = [out]
    if not force and out.exists():
        return {"sample_id": sid, "T": T_name, "status": "skipped_exists", "paths": [str(out)]}

    v1_path = sample.get("_user_1_image") or sample.get("user_1_image_local_path")
    v2_path = sample.get("_user_2_image") or sample.get("user_2_image_local_path")
    if not v1_path or not v2_path:
        return {"sample_id": sid, "T": T_name, "status": "error", "error": "v1/v2 path missing"}

    try:
        img1_bytes = encode_png_bytes(v1_path)
        img2_bytes = encode_png_bytes(v2_path)
    except FileNotFoundError as e:
        return {"sample_id": sid, "T": T_name, "status": "error", "error": f"input image not found: {e}"}

    img1_part = types.Part.from_bytes(data=img1_bytes, mime_type="image/png")
    img2_part = types.Part.from_bytes(data=img2_bytes, mime_type="image/png")
    contents = [PROMPTS[T_name], img1_part, img2_part]

    try:
        response = call_with_retry(client, model, contents)
    except Exception as e:
        return {"sample_id": sid, "T": T_name, "status": "error", "error": f"generate_content: {e}"}

    images = extract_images(response)
    if not images:
        text = ""
        try:
            text = response.text or ""
        except Exception:
            pass
        return {"sample_id": sid, "T": T_name, "status": "no_images", "refusal_text": text[:300]}

    images[0].save(outs[0])
    return {"sample_id": sid, "T": T_name, "status": "ok", "paths": [str(outs[0])],
            "n_returned": len(images)}


# Load V1/V2 image paths from the COSMIC test JSONs.  The provenance index
# rows don't carry them.
TEST_JSONS = {
    "anchor": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
    "counting": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
    "relative_distance": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
    "relative_direction": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
}


def hydrate_image_paths(samples):
    """Attach _user_1_image / _user_2_image to each sample by looking up
    the COSMIC test JSON for its subtask + sample_id."""
    by_subtask = {}
    for st, jp in TEST_JSONS.items():
        d = {x["sample_id"]: x for x in json.load(open(jp))}
        by_subtask[st] = d
    for s in samples:
        st = s["subtask"]
        raw = by_subtask.get(st, {}).get(s["sample_id"])
        if raw:
            s["_user_1_image"] = raw["user_1_image_local_path"]
            s["_user_2_image"] = raw["user_2_image_local_path"]
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/gemini_oracles",
    )
    parser.add_argument("--model", default="gemini-3-pro-image-preview")
    parser.add_argument(
        "--views",
        nargs="+",
        choices=["topdown", "panorama", "cor"],
        default=["topdown", "panorama", "cor"],
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap the sample count for smoke runs.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log_path", default=None,
                        help="JSONL log of per-call statuses (default: <output_dir>/_log.jsonl).")
    args = parser.parse_args()

    api_key = os.environ.get("VisualCoT_GEMINI")
    if not api_key:
        sys.exit("VisualCoT_GEMINI env var not set (source ~/.bashrc).")
    client = genai.Client(api_key=api_key)
    print(f"Gemini client: model={args.model}")

    samples = load_sample_ids_from_audit()
    samples = hydrate_image_paths(samples)
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"Loaded {len(samples)} samples across {len(set(s['subtask'] for s in samples))} subtasks")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_path) if args.log_path else (out_root / "_log.jsonl")
    log_fp = open(log_path, "a")

    n_tasks = len(samples) * len(args.views)
    print(f"Tasks: {n_tasks}  workers={args.workers}")

    counters = {"ok": 0, "ok_partial": 0, "skipped_exists": 0, "no_images": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for s in samples:
            for v in args.views:
                futs.append(ex.submit(generate_one, client, args.model, s, v, str(out_root), args.force))
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            log_fp.write(json.dumps(row) + "\n")
            log_fp.flush()
            st = row.get("status", "error")
            counters[st] = counters.get(st, 0) + 1
            if i % 10 == 0 or i == n_tasks:
                print(f"  {i}/{n_tasks}  " + "  ".join(f"{k}={v}" for k, v in counters.items()), flush=True)

    log_fp.close()
    print("done.")
    print("summary:", counters)
    print("log:", log_path)


if __name__ == "__main__":
    main()
