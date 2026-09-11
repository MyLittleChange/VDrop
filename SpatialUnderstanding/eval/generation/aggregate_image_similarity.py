#!/usr/bin/env python3
"""
Aggregate Two-Reader image-similarity numbers:
  pixel    : DINO cos / LPIPS / PSNR / SSIM (one JSON per (pairing, subtask))
  gemini   : 1-5 score from Gemini-Flash autorater (one JSON per (pairing, subtask))

Writes a single Markdown + CSV table.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

PAIRINGS = ["pano", "cor", "td", "cor_view"]
PAIRING_LABELS = {
    "pano":     "Panorama (T_pano vs T_pano_gen)",
    "cor":      "Correspondence (T_cor vs T_cor_gen)",
    "td":       "Top-down (T_td_blender vs T_td_gen)",
    "cor_view": "Corner-view (T_cor_view vs T_cor_view_gen)",
}
SUBTASKS = ["anchor", "counting", "relative_distance", "relative_direction"]
ROOM_TYPES = ["LivingRoom", "Bedroom", "Kitchen", "DiningRoom", "Bathroom"]

DEFAULT_ROOT = "/path/to/scratch/VisualCoT/eval_results/image_similarity"
SUBSET_DIR = "/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/eval_subset_200"


def _load_sample_metadata():
    """Return {sample_id: {room_type, question_type, subtask}} from approved_mcqs."""
    meta = {}
    for st in SUBTASKS:
        p = f"/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_{st}_normalized.json"
        d = json.load(open(p))
        for s in d:
            sid = s["sample_id"]
            rp = s.get("room_part","")
            room = rp.split("_")[0] if rp else "unknown"
            # Normalize BedRoom -> Bedroom
            if room == "BedRoom":
                room = "Bedroom"
            meta[sid] = {
                "room_type": room,
                "question_type": s.get("question_type", ""),
                "subtask": st,
            }
    return meta


def _mean(values, key=None):
    vals = [v if key is None else v.get(key) for v in values]
    vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    return (sum(vals) / len(vals)) if vals else float("nan")


def load_pixel(root: Path, pairing: str, subtask: str) -> List[Dict]:
    p = root / "pixel" / pairing / f"{subtask}.json"
    if not p.exists():
        return []
    return json.load(open(p)).get("results", [])


def load_gemini(root: Path, pairing: str, subtask: str) -> List[Dict]:
    p = root / "gemini_judge" / pairing / f"{subtask}.json"
    if not p.exists():
        return []
    return json.load(open(p)).get("results", [])


def _filter(rows: List[Dict], allowed_sids: set) -> List[Dict]:
    if allowed_sids is None:
        return rows
    return [r for r in rows if r.get("sample_id") in allowed_sids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--paired", action="store_true",
                        help="Restrict every metric to the sample_ids common to "
                             "all 3 pairings (clean apples-to-apples comparison).")
    args = parser.parse_args()
    root = Path(args.root)

    # Build the paired intersection if requested
    allowed_sids = None
    if args.paired:
        sets = []
        for pairing in PAIRINGS:
            ids = set()
            for st in SUBTASKS:
                for r in load_pixel(root, pairing, st):
                    ids.add(r["sample_id"])
            sets.append(ids)
        allowed_sids = set.intersection(*sets) if sets else set()
        print(f"[paired] {len(allowed_sids)} sample_ids in common across all pairings")

    # Sample metadata for sub-category breakdowns
    meta = _load_sample_metadata()

    # Collect rows
    rows = {}  # (pairing, subtask) -> dict of metrics + counts
    for pairing in PAIRINGS:
        for subtask in SUBTASKS:
            pix = _filter(load_pixel(root, pairing, subtask), allowed_sids)
            jud = _filter(load_gemini(root, pairing, subtask), allowed_sids)
            n_pix = len(pix)
            n_jud = sum(1 for r in jud if r.get("status") == "ok")
            rows[(pairing, subtask)] = {
                "n_pix": n_pix,
                "n_jud": n_jud,
                "siglip": _mean(pix, "siglip_cos"),
                "dino": _mean(pix, "dino_cos"),
                "lpips": _mean(pix, "lpips"),
                "psnr": _mean(pix, "psnr"),
                "ssim": _mean(pix, "ssim"),
                "gemini": _mean([r for r in jud if r.get("status") == "ok"], "score"),
                "gemini_dist": Counter(r.get("score") for r in jud if r.get("status") == "ok"),
            }

    # Per-pairing overall
    overall = {}
    for pairing in PAIRINGS:
        agg_pix = []; agg_jud = []
        for subtask in SUBTASKS:
            agg_pix.extend(_filter(load_pixel(root, pairing, subtask), allowed_sids))
            agg_jud.extend([r for r in _filter(load_gemini(root, pairing, subtask), allowed_sids) if r.get("status") == "ok"])
        overall[pairing] = {
            "n_pix": len(agg_pix),
            "n_jud": len(agg_jud),
            "siglip": _mean(agg_pix, "siglip_cos"),
            "dino": _mean(agg_pix, "dino_cos"),
            "lpips": _mean(agg_pix, "lpips"),
            "psnr": _mean(agg_pix, "psnr"),
            "ssim": _mean(agg_pix, "ssim"),
            "gemini": _mean(agg_jud, "score"),
            "gemini_dist": Counter(r.get("score") for r in agg_jud),
        }

    # Per-pairing × room_type breakdown
    by_room = {}  # (pairing, room) -> metrics
    for pairing in PAIRINGS:
        for room in ROOM_TYPES:
            agg_pix = []; agg_jud = []
            for subtask in SUBTASKS:
                pix_all = _filter(load_pixel(root, pairing, subtask), allowed_sids)
                jud_all = _filter(load_gemini(root, pairing, subtask), allowed_sids)
                pix_room = [r for r in pix_all if meta.get(r["sample_id"], {}).get("room_type") == room]
                jud_room = [r for r in jud_all if meta.get(r["sample_id"], {}).get("room_type") == room and r.get("status") == "ok"]
                agg_pix.extend(pix_room)
                agg_jud.extend(jud_room)
            by_room[(pairing, room)] = {
                "n_pix": len(agg_pix),
                "n_jud": len(agg_jud),
                "siglip": _mean(agg_pix, "siglip_cos"),
                "dino": _mean(agg_pix, "dino_cos"),
                "lpips": _mean(agg_pix, "lpips"),
                "psnr": _mean(agg_pix, "psnr"),
                "ssim": _mean(agg_pix, "ssim"),
                "gemini": _mean(agg_jud, "score"),
            }

    # Per-pairing × relative_distance fine question_type (closest/farthest)
    rd_qtypes = ["closest", "farthest"]
    by_rd_qtype = {}
    for pairing in PAIRINGS:
        for qt in rd_qtypes:
            pix_all = _filter(load_pixel(root, pairing, "relative_distance"), allowed_sids)
            jud_all = _filter(load_gemini(root, pairing, "relative_distance"), allowed_sids)
            pix_qt = [r for r in pix_all if meta.get(r["sample_id"], {}).get("question_type") == qt]
            jud_qt = [r for r in jud_all if meta.get(r["sample_id"], {}).get("question_type") == qt and r.get("status") == "ok"]
            by_rd_qtype[(pairing, qt)] = {
                "n_pix": len(pix_qt),
                "n_jud": len(jud_qt),
                "siglip": _mean(pix_qt, "siglip_cos"),
                "dino": _mean(pix_qt, "dino_cos"),
                "lpips": _mean(pix_qt, "lpips"),
                "psnr": _mean(pix_qt, "psnr"),
                "ssim": _mean(pix_qt, "ssim"),
                "gemini": _mean(jud_qt, "score"),
            }

    # Print overall
    print(f"\n{'pairing':35}{'N(pix)':>8}{'N(jud)':>8}{'SigLIP↑':>10}{'DINO↑':>10}{'LPIPS↓':>10}{'PSNR↑':>10}{'SSIM↑':>10}{'Gem↑':>10}")
    print("-" * 101)
    for pairing in PAIRINGS:
        o = overall[pairing]
        print(f"{PAIRING_LABELS[pairing]:35}{o['n_pix']:>8}{o['n_jud']:>8}{o['siglip']:>10.4f}{o['dino']:>10.4f}{o['lpips']:>10.4f}{o['psnr']:>10.2f}{o['ssim']:>10.4f}{o['gemini']:>10.2f}")

    # Print per-subtask
    print("\nPer-subtask breakdown:")
    for pairing in PAIRINGS:
        print(f"\n  {PAIRING_LABELS[pairing]}")
        print(f"    {'subtask':22}{'N':>6}{'SigLIP↑':>10}{'DINO↑':>10}{'LPIPS↓':>10}{'PSNR↑':>10}{'SSIM↑':>10}{'Gem↑':>10}")
        for subtask in SUBTASKS:
            r = rows[(pairing, subtask)]
            print(f"    {subtask:22}{r['n_pix']:>6}{r['siglip']:>10.4f}{r['dino']:>10.4f}{r['lpips']:>10.4f}{r['psnr']:>10.2f}{r['ssim']:>10.4f}{r['gemini']:>10.2f}")

    # Print by room_type
    print("\nBy room type (sub-category breakdown):")
    for pairing in PAIRINGS:
        print(f"\n  {PAIRING_LABELS[pairing]}")
        print(f"    {'room_type':22}{'N':>6}{'SigLIP↑':>10}{'DINO↑':>10}{'LPIPS↓':>10}{'PSNR↑':>10}{'SSIM↑':>10}{'Gem↑':>10}")
        for room in ROOM_TYPES:
            r = by_room.get((pairing, room), {})
            if r.get("n_pix", 0) == 0:
                continue
            print(f"    {room:22}{r['n_pix']:>6}{r['siglip']:>10.4f}{r['dino']:>10.4f}{r['lpips']:>10.4f}{r['psnr']:>10.2f}{r['ssim']:>10.4f}{r['gemini']:>10.2f}")

    # Print rel-distance closest/farthest
    print("\nBy relative_distance question_type (closest vs farthest):")
    for pairing in PAIRINGS:
        print(f"\n  {PAIRING_LABELS[pairing]}")
        print(f"    {'question_type':22}{'N':>6}{'SigLIP↑':>10}{'DINO↑':>10}{'LPIPS↓':>10}{'PSNR↑':>10}{'SSIM↑':>10}{'Gem↑':>10}")
        for qt in ["closest", "farthest"]:
            r = by_rd_qtype.get((pairing, qt), {})
            if r.get("n_pix", 0) == 0:
                continue
            print(f"    {qt:22}{r['n_pix']:>6}{r['siglip']:>10.4f}{r['dino']:>10.4f}{r['lpips']:>10.4f}{r['psnr']:>10.2f}{r['ssim']:>10.4f}{r['gemini']:>10.2f}")

    # Print Gemini score distributions
    print("\nGemini 1-5 score distribution (overall, per pairing):")
    for pairing in PAIRINGS:
        d = overall[pairing]["gemini_dist"]
        n = sum(d.values())
        if n == 0:
            print(f"  {PAIRING_LABELS[pairing]:35}: (no judge data)")
            continue
        pct = {k: 100 * d.get(k, 0) / n for k in [1, 2, 3, 4, 5]}
        bar = "  ".join(f"{k}:{pct[k]:>5.1f}%" for k in [1, 2, 3, 4, 5])
        print(f"  {PAIRING_LABELS[pairing]:35}: {bar}")

    # Save CSV
    out_csv = root / "image_similarity_table.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pairing","subtask","n_pix","n_jud","siglip_cos","dino_cos","lpips","psnr","ssim","gemini_mean"])
        for pairing in PAIRINGS:
            for subtask in SUBTASKS:
                r = rows[(pairing, subtask)]
                w.writerow([pairing, subtask, r['n_pix'], r['n_jud'],
                            f"{r['siglip']:.4f}", f"{r['dino']:.4f}", f"{r['lpips']:.4f}",
                            f"{r['psnr']:.2f}", f"{r['ssim']:.4f}", f"{r['gemini']:.4f}"])
        for pairing in PAIRINGS:
            o = overall[pairing]
            w.writerow([pairing, "overall", o['n_pix'], o['n_jud'],
                        f"{o['siglip']:.4f}", f"{o['dino']:.4f}", f"{o['lpips']:.4f}",
                        f"{o['psnr']:.2f}", f"{o['ssim']:.4f}", f"{o['gemini']:.4f}"])
    print(f"\nWrote {out_csv}")

    # Save Markdown
    out_md = root / "image_similarity_table.md"
    with open(out_md, "w") as f:
        f.write("# Two-Reader: Direct Generated-vs-GT Image Similarity\n\n")
        f.write("## Setting\n\n")
        if args.paired:
            f.write(f"**Sample restriction**: paired set of **{len(allowed_sids) if allowed_sids else 0}** sample_ids common across all 3 pairings ({{pano, cor, td}}). Every cell in every table below uses this same denominator, so all comparisons are apples-to-apples. The 116-sample gap relative to the 837-sample eval subset is the same missing-`.blend` set that affects T_pano/T_td_blender (see PROGRESS.md §12).\n\n")
        else:
            f.write("**Sample restriction**: no paired restriction; each pairing uses every sample it has on disk. Use `--paired` for an apples-to-apples comparison.\n\n")
        f.write("**Pairings** (each compares a GT bridge image against the BAGEL bridge-masked checkpoint's generated counterpart):\n\n")
        f.write("- **Panorama**: GT 360° Cycles render vs. BAGEL output from `mix_all_balance_visual_only_bridge_masked_lora_7k`.\n")
        f.write("- **Correspondence**: GT side-by-side V1\\|V2 composite with coloured-dot correspondences vs. BAGEL output from `pm_no_rotation_visual_only_bridge_masked_lora_7k`. **NB**: GT is 2560×720 (stitched pair), generated is 1024×720 (single image); LPIPS/PSNR/SSIM are penalised by this aspect mismatch — DINO and the Gemini judge are the meaningful metrics here.\n")
        f.write("- **Top-down**: GT 1024×1024 photoreal Cycles ortho vs. BAGEL output from `topdown_round3_visual_only_bridge_masked_lora_7k`.\n\n")
        f.write("**Metrics**:\n\n")
        f.write("- **SigLIP cosine similarity** (primary, semantic, aspect-preserving): `google/siglip-large-patch16-384` vision-tower pooled output. Images are pad-to-square letterboxed *before* SigLIP's processor resizes them to 384×384, so the full FoV of wide GT panoramas / stitched composites contributes to the comparison. Higher is better.\n")
        f.write("- **DINOv2 cosine similarity** (secondary, semantic): `facebook/dinov2-base` CLS token. Uses HF's default resize+center-crop, which silently throws away wide GT content. Kept for comparison with prior reports. Higher is better.\n")
        f.write("- **LPIPS-alex** (perceptual): resizes GT to 1024×720 to match generated. Lower is better.\n")
        f.write("- **PSNR / SSIM**: pixel-level, also after resize. Higher is better.\n")
        f.write("- **Gemini-3-Flash autorater**: 1–5 similarity score (5 = near-identical, 1 = unrelated) with a one-sentence rationale per pair. Note: the Gemini rubric anchors at *fidelity*, so it favours pairings that are easier to generate (e.g. correspondence is just restyled V1) over pairings that synthesise novel views (panorama/top-down). Use as a diagnostic, not a clean cross-pairing ranking.\n\n")
        f.write("## Overall (per-pairing)\n\n")
        f.write("| Pairing | N (pix) | N (judge) | SigLIP cos ↑ | DINO cos ↑ | LPIPS ↓ | PSNR ↑ | SSIM ↑ | Gemini 1-5 ↑ |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for pairing in PAIRINGS:
            o = overall[pairing]
            f.write(f"| {PAIRING_LABELS[pairing]} | {o['n_pix']} | {o['n_jud']} | {o['siglip']:.4f} | {o['dino']:.4f} | {o['lpips']:.4f} | {o['psnr']:.2f} | {o['ssim']:.4f} | {o['gemini']:.2f} |\n")

        f.write("\n## Per-subtask breakdown\n\n")
        for pairing in PAIRINGS:
            f.write(f"### {PAIRING_LABELS[pairing]}\n\n")
            f.write("| Subtask | N | SigLIP cos ↑ | DINO cos ↑ | LPIPS ↓ | PSNR ↑ | SSIM ↑ | Gemini 1-5 ↑ |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for subtask in SUBTASKS:
                r = rows[(pairing, subtask)]
                f.write(f"| {subtask} | {r['n_pix']} | {r['siglip']:.4f} | {r['dino']:.4f} | {r['lpips']:.4f} | {r['psnr']:.2f} | {r['ssim']:.4f} | {r['gemini']:.2f} |\n")
            f.write("\n")

        f.write("## By room type (sub-category breakdown)\n\n")
        for pairing in PAIRINGS:
            f.write(f"### {PAIRING_LABELS[pairing]}\n\n")
            f.write("| Room type | N | SigLIP cos ↑ | DINO cos ↑ | LPIPS ↓ | PSNR ↑ | SSIM ↑ | Gemini 1-5 ↑ |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for room in ROOM_TYPES:
                r = by_room.get((pairing, room), {})
                if r.get("n_pix", 0) == 0:
                    continue
                f.write(f"| {room} | {r['n_pix']} | {r['siglip']:.4f} | {r['dino']:.4f} | {r['lpips']:.4f} | {r['psnr']:.2f} | {r['ssim']:.4f} | {r['gemini']:.2f} |\n")
            f.write("\n")

        # rel-distance qtype (only meaningful for that subtask)
        any_qt = any(by_rd_qtype.get((p, qt), {}).get("n_pix", 0) > 0 for p in PAIRINGS for qt in ["closest", "farthest"])
        if any_qt:
            f.write("## Relative-distance `closest` vs `farthest` (sub-category)\n\n")
            for pairing in PAIRINGS:
                f.write(f"### {PAIRING_LABELS[pairing]}\n\n")
                f.write("| Question type | N | SigLIP cos ↑ | DINO cos ↑ | LPIPS ↓ | PSNR ↑ | SSIM ↑ | Gemini 1-5 ↑ |\n")
                f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
                for qt in ["closest", "farthest"]:
                    r = by_rd_qtype.get((pairing, qt), {})
                    if r.get("n_pix", 0) == 0:
                        continue
                    f.write(f"| {qt} | {r['n_pix']} | {r['siglip']:.4f} | {r['dino']:.4f} | {r['lpips']:.4f} | {r['psnr']:.2f} | {r['ssim']:.4f} | {r['gemini']:.2f} |\n")
                f.write("\n")

        f.write("## Gemini 1-5 score distribution\n\n")
        f.write("| Pairing | 1 | 2 | 3 | 4 | 5 |\n|---|---:|---:|---:|---:|---:|\n")
        for pairing in PAIRINGS:
            d = overall[pairing]["gemini_dist"]
            n = sum(d.values())
            if n == 0:
                f.write(f"| {PAIRING_LABELS[pairing]} | — | — | — | — | — |\n")
                continue
            f.write(f"| {PAIRING_LABELS[pairing]} |")
            for k in [1, 2, 3, 4, 5]:
                pct = 100 * d.get(k, 0) / n
                f.write(f" {pct:.1f}% ({d.get(k,0)}) |")
            f.write("\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
