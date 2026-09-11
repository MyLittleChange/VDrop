#!/usr/bin/env python3
"""
Build the Gemini-pilot eval subset (100 samples — 25 per subtask) for the
Two-Reader Oracle Ablation. Same 100 samples that the Gemini oracle
generator produced renders for.

Inlines GT oracle paths AND the Gemini-generated paths so the Qwen oracle
inference script can pick whichever condition we run.

Output:
  /path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/eval_subset_gemini_100/<subtask>.json
"""
import json
from pathlib import Path

from PIL import Image

PROVENANCE_INDEX = "/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/two_reader_informativeness_index.json"
TEST_JSONS = {
    "anchor": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
    "counting": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
    "relative_distance": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
    "relative_direction": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
}
OUTPUT_DIR = Path("/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/eval_subset_gemini_100")
COR_COMPOSITE_DIR = Path("/path/to/scratch/VisualCoT/infinigen/correspondence_shared_objects_approved_mcqs")
COR_GEMINI_COMPOSITE_DIR = Path("/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/gemini_oracles/cor_composites")


def make_composite(cor_paths, out_path):
    """Stitch two images horizontally into a single JPEG.
    Returns the output path string, or None if either input is missing.
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

    idx = json.load(open(PROVENANCE_INDEX))
    by_sid = {r["sample_id"]: r for r in idx["samples"]}

    # 100 samples that have Gemini renders (from filenames)
    pilot_sids = set()
    for p in (Path("/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/gemini_oracles/topdown").rglob("topdown_gemini_*.png")):
        pilot_sids.add(p.stem.replace("topdown_gemini_", ""))
    print(f"Gemini pilot sample_ids: {len(pilot_sids)}")

    grand = {"total": 0, "per_subtask": {}}
    for subtask, jp in TEST_JSONS.items():
        all_samples = json.load(open(jp))
        kept = [s for s in all_samples if s["sample_id"] in pilot_sids]

        cov = {k: 0 for k in (
            "T_td_blender", "T_pano", "T_cor_composite", "T_noise",
            "T_td_gemini", "T_pano_gemini", "T_cor_gemini_composite",
        )}

        for s in kept:
            sid = s["sample_id"]
            prov = by_sid.get(sid, {})

            s["_T_td_blender"] = prov.get("T_td_blender")
            s["_T_pano"] = prov.get("T_pano")
            s["_T_noise"] = prov.get("T_noise")

            t_cor = prov.get("T_cor")
            if t_cor:
                comp = COR_COMPOSITE_DIR / s["scene_id"] / "composite_v1v2.jpg"
                if comp.exists():
                    s["_T_cor_composite"] = str(comp)
                else:
                    s["_T_cor_composite"] = make_composite(t_cor, comp)
            else:
                s["_T_cor_composite"] = None

            s["_T_td_gemini"] = prov.get("T_td_gemini")
            s["_T_pano_gemini"] = prov.get("T_pano_gemini")

            cor_g = prov.get("T_cor_gemini")
            if cor_g:
                comp_g = COR_GEMINI_COMPOSITE_DIR / s["scene_id"] / f"composite_{sid}.jpg"
                if comp_g.exists():
                    s["_T_cor_gemini_composite"] = str(comp_g)
                else:
                    s["_T_cor_gemini_composite"] = make_composite(cor_g, comp_g)
            else:
                s["_T_cor_gemini_composite"] = None

            for k in cov:
                if s.get(f"_{k}"):
                    cov[k] += 1

        out_path = OUTPUT_DIR / f"{subtask}.json"
        json.dump(kept, open(out_path, "w"), indent=2)
        print(f"[{subtask}] kept={len(kept)} | coverage: {cov}")
        grand["per_subtask"][subtask] = {"n_kept": len(kept), "oracle_coverage": cov}
        grand["total"] += len(kept)

    print(f"\nTotal: {grand['total']}")
    json.dump(grand, open(OUTPUT_DIR / "_summary.json", "w"), indent=2)
    print(f"Wrote subset JSONs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
