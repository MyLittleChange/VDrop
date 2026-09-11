#!/usr/bin/env python3
"""
Build the 200-per-category eval subset for the Two-Reader Oracle Ablation.

For each of the 4 COSMIC subtasks:
  1. Deterministically subsample 200/250 samples by hash(sample_id).
  2. Inline-merge the oracle paths from the provenance index
     (T_td_blender / T_cor / T_pano / T_noise) into each retained sample.
  3. For T_cor (two annotated images), pre-build a side-by-side composite
     PNG per scene and add `_T_cor_composite` field.

Outputs:
  /path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/eval_subset_200/<subtask>.json

Same 200 sample_ids retained across all 4 conditions (the condition
just picks a different inlined field at inference time).
"""
import hashlib
import json
from pathlib import Path

from PIL import Image

TEST_JSONS = {
    "anchor": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
    "counting": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
    "relative_distance": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
    "relative_direction": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
}

PROVENANCE_INDEX = "/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/two_reader_informativeness_index.json"
OUTPUT_DIR = Path("/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/eval_subset_200")
COR_COMPOSITE_DIR = Path("/path/to/scratch/VisualCoT/infinigen/correspondence_shared_objects_approved_mcqs")

# Model-generated bridge images from 3 BAGEL bridge-masked checkpoints.
# Output naming: <eval_output_dir>/generated_images/<sample_id>_round_0.png
# (set in inference/run_inference_bagel_spatial.py:478-480).
GEN_CKPTS = {
    "_T_pano_gen": "BAGEL_format_mix_all_balance_visual_only_bridge_masked_lora_7k",
    "_T_cor_gen":  "BAGEL_format_pm_no_rotation_visual_only_bridge_masked_lora_7k",
    "_T_td_gen":   "BAGEL_format_topdown_round3_visual_only_bridge_masked_lora_7k",
    "_T_cor_view_gen": "BAGEL_format_corner_view_visual_only_bridge_masked_lora_7k",
}
GEN_BASE = Path("/path/to/scratch/VisualCoT")

KEEP_THRESHOLD = 200  # out of 250


def should_keep(sample_id: str) -> bool:
    """Deterministic SHA-256 subsample: keep 200/250 (~80%)."""
    h = int(hashlib.sha256(sample_id.encode()).hexdigest(), 16)
    return (h % 250) < KEEP_THRESHOLD


def make_cor_composite(cor_paths, out_path):
    """Stitch annotated_cam0 | annotated_cam1 horizontally into one JPEG.

    JPEG (q=92) instead of PNG: ~500x faster encode, ~8x smaller file,
    no perceptible quality loss for this audit task.
    """
    if out_path.exists():
        return str(out_path)
    if not cor_paths or len(cor_paths) != 2:
        return None
    p0, p1 = cor_paths
    if not (Path(p0).exists() and Path(p1).exists()):
        return None
    im0 = Image.open(p0).convert("RGB")
    im1 = Image.open(p1).convert("RGB")
    # Match heights by resizing the taller one down
    h = min(im0.height, im1.height)
    if im0.height != h:
        im0 = im0.resize((int(im0.width * h / im0.height), h))
    if im1.height != h:
        im1 = im1.resize((int(im1.width * h / im1.height), h))
    composite = Image.new("RGB", (im0.width + im1.width, h))
    composite.paste(im0, (0, 0))
    composite.paste(im1, (im0.width, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_path, "JPEG", quality=92)
    return str(out_path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load provenance index keyed by sample_id
    idx = json.load(open(PROVENANCE_INDEX))
    by_sid = {r["sample_id"]: r for r in idx["samples"]}

    grand = {"total_kept": 0, "per_subtask": {}}
    for subtask, jp in TEST_JSONS.items():
        all_samples = json.load(open(jp))
        kept = [s for s in all_samples if should_keep(s["sample_id"])]
        if len(kept) != KEEP_THRESHOLD:
            print(f"  WARNING: {subtask} kept {len(kept)}, expected {KEEP_THRESHOLD}")

        # Inline-merge oracle paths
        coverage = {"T_td_blender": 0, "T_cor_composite": 0, "T_pano": 0, "T_noise": 0,
                    "T_pano_gen": 0, "T_cor_gen": 0, "T_td_gen": 0, "T_cor_view_gen": 0}
        for s in kept:
            sid = s["sample_id"]
            prov = by_sid.get(sid, {})
            s["_T_td_blender"] = prov.get("T_td_blender")
            s["_T_pano"] = prov.get("T_pano")
            s["_T_noise"] = prov.get("T_noise")

            # Model-generated bridge images (per-checkpoint, per-subtask).
            for field, ckpt in GEN_CKPTS.items():
                gen_path = GEN_BASE / f"{ckpt}_mcqs_{subtask}_normalized" / "generated_images" / f"{sid}_round_0.png"
                s[field] = str(gen_path) if gen_path.exists() else None

            # Build T_cor composite per scene (JPEG; legacy .png reused if present)
            t_cor = prov.get("T_cor")
            if t_cor:
                jpg_path = COR_COMPOSITE_DIR / s["scene_id"] / "composite_v1v2.jpg"
                png_path = COR_COMPOSITE_DIR / s["scene_id"] / "composite_v1v2.png"
                if jpg_path.exists():
                    s["_T_cor_composite"] = str(jpg_path)
                elif png_path.exists():
                    s["_T_cor_composite"] = str(png_path)  # legacy from earlier run
                else:
                    s["_T_cor_composite"] = make_cor_composite(t_cor, jpg_path)
            else:
                s["_T_cor_composite"] = None

            for k in coverage:
                if s.get(f"_{k}"):
                    coverage[k] += 1

        out_path = OUTPUT_DIR / f"{subtask}.json"
        json.dump(kept, open(out_path, "w"), indent=2)
        print(f"[{subtask}] kept={len(kept)} | coverage: {coverage}")
        grand["per_subtask"][subtask] = {"n_kept": len(kept), "oracle_coverage": coverage}
        grand["total_kept"] += len(kept)

    print(f"\nTotal kept: {grand['total_kept']}")
    print(f"Wrote subset JSONs to {OUTPUT_DIR}")

    # Also write a summary
    summary_path = OUTPUT_DIR / "_summary.json"
    json.dump(grand, open(summary_path, "w"), indent=2)


if __name__ == "__main__":
    main()
