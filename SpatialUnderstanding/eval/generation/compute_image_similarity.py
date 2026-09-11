#!/usr/bin/env python3
"""
Direct generated-vs-GT image-similarity eval for the Two-Reader Informativeness
pipeline. Computes per-pair similarity for each (GT, BAGEL-generated) pairing:

    pano : T_pano (GT 360° panorama)       vs T_pano_gen (BAGEL bridge-masked output)
    cor  : T_cor  (GT V1|V2 dot composite) vs T_cor_gen
    td   : T_td_blender (GT Cycles ortho)  vs T_td_gen

Primary metric: **DINOv2 cosine similarity** (semantic, robust to resolution
and aspect-ratio mismatches our pairings suffer from).
Secondary: LPIPS-alex / PSNR / SSIM for completeness (resize GT to match
generated). Outputs one JSON per (pairing, subtask).

Reads the eval subset at:
    /path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/eval_subset_200/<subtask>.json
Per-row writes to:
    /path/to/scratch/VisualCoT/eval_results/image_similarity/pixel/<pairing>/<subtask>.json
"""
import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ─────────────────────── configuration ───────────────────────

PAIRINGS = {
    "pano":     ("_T_pano",          "_T_pano_gen"),
    "cor":      ("_T_cor_composite", "_T_cor_gen"),
    "td":       ("_T_td_blender",    "_T_td_gen"),
    "cor_view": ("_T_cor_view",      "_T_cor_view_gen"),
}
SUBTASKS = ("anchor", "counting", "relative_distance", "relative_direction")

DEFAULT_SUBSET_DIR = "/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/eval_subset_200"
DEFAULT_OUTPUT_DIR = "/path/to/scratch/VisualCoT/eval_results/image_similarity/pixel"

# Resize target for LPIPS/PSNR/SSIM. Matches the generated images.
PIXEL_TARGET_HW = (720, 1024)  # (H, W) — torch convention is (..., H, W)


# ─────────────────────── preprocessing helpers ───────────────────────

def pad_to_square(im: Image.Image, fill=(0, 0, 0)) -> Image.Image:
    """Letterbox an image to a square with black padding. Preserves aspect."""
    if im.mode != "RGB":
        im = im.convert("RGB")
    w, h = im.size
    if w == h:
        return im
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return canvas


# ─────────────────────── SigLIP (aspect-preserving) ───────────────────────

class SiglipSimilarity:
    """SigLIP image-image cosine similarity with pad-to-square preprocessing.

    Pad-to-square FIRST (so aspect ratio is preserved), then let SigLIP's
    own processor resize the square to its native size (384×384 for the
    large model). This way both GT and generated images contribute their
    full content to the comparison, regardless of aspect mismatch.
    """

    def __init__(self, model_name: str = "google/siglip-large-patch16-384", device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModel
        self.device = device
        print(f"[siglip] loading {model_name} on {device} ...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.model_name = model_name

    @torch.no_grad()
    def encode_batch(self, pil_images: List[Image.Image]) -> torch.Tensor:
        squares = [pad_to_square(im) for im in pil_images]
        inputs = self.processor(images=squares, return_tensors="pt").to(self.device)
        # Vision tower only — pool over patches and skip the text branch.
        out = self.model.vision_model(**inputs)
        # SigLIP uses pooler_output (mean-pooled or attention-pooled
        # depending on variant; both are valid scene-level features).
        feat = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state.mean(dim=1)
        feat = F.normalize(feat, dim=-1)
        return feat

    @torch.no_grad()
    def cos_similarity(self, gt_images: List[Image.Image], gen_images: List[Image.Image]) -> List[float]:
        assert len(gt_images) == len(gen_images)
        feat_gt = self.encode_batch(gt_images)
        feat_gen = self.encode_batch(gen_images)
        sims = (feat_gt * feat_gen).sum(dim=-1)
        return [float(x) for x in sims.cpu().tolist()]


# ─────────────────────── DINOv2 ───────────────────────

class DinoSimilarity:
    """Encodes both images, computes cosine similarity on CLS token."""

    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModel
        self.device = device
        print(f"[dino] loading {model_name} on {device} ...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.model_name = model_name

    @torch.no_grad()
    def encode_batch(self, pil_images: List[Image.Image]) -> torch.Tensor:
        """Returns (B, D) CLS features, L2-normalized."""
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        cls = out.last_hidden_state[:, 0, :]
        cls = F.normalize(cls, dim=-1)
        return cls

    @torch.no_grad()
    def cos_similarity(self, gt_images: List[Image.Image], gen_images: List[Image.Image]) -> List[float]:
        assert len(gt_images) == len(gen_images)
        feat_gt = self.encode_batch(gt_images)
        feat_gen = self.encode_batch(gen_images)
        sims = (feat_gt * feat_gen).sum(dim=-1)
        return [float(x) for x in sims.cpu().tolist()]


# ─────────────────────── LPIPS / PSNR / SSIM ───────────────────────

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def compute_ssim_simple(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Plain SSIM via mean/variance over the full image. Fast, no kernel."""
    C1, C2 = (0.01) ** 2, (0.03) ** 2
    mu_p = pred.mean(dim=[2, 3], keepdim=True)
    mu_t = target.mean(dim=[2, 3], keepdim=True)
    var_p = ((pred - mu_p) ** 2).mean(dim=[2, 3], keepdim=True)
    var_t = ((target - mu_t) ** 2).mean(dim=[2, 3], keepdim=True)
    cov = ((pred - mu_p) * (target - mu_t)).mean(dim=[2, 3], keepdim=True)
    num = (2 * mu_p * mu_t + C1) * (2 * cov + C2)
    den = (mu_p ** 2 + mu_t ** 2 + C1) * (var_p + var_t + C2)
    return float((num / den).mean().item())


class PixelMetrics:
    """LPIPS-alex (per-pair) plus PSNR + simple SSIM."""

    def __init__(self, device: str = "cuda", lpips_net: str = "alex"):
        import lpips
        self.device = device
        print(f"[lpips] loading lpips-{lpips_net} on {device} ...")
        self.lpips = lpips.LPIPS(net=lpips_net).to(device).eval()

    @torch.no_grad()
    def compute(self, gt_pil: Image.Image, gen_pil: Image.Image) -> Tuple[float, float, float]:
        """Return (lpips, psnr, ssim). Resize GT to match (1024, 720)."""
        H, W = PIXEL_TARGET_HW
        gt = gt_pil.convert("RGB").resize((W, H), Image.BICUBIC)
        gen = gen_pil.convert("RGB").resize((W, H), Image.BICUBIC) if gen_pil.size != (W, H) else gen_pil.convert("RGB")
        import torchvision.transforms.functional as TF
        gt_t = TF.to_tensor(gt).unsqueeze(0).to(self.device)
        gen_t = TF.to_tensor(gen).unsqueeze(0).to(self.device)
        # LPIPS expects [-1, 1]
        lpips_val = float(self.lpips(2 * gt_t - 1, 2 * gen_t - 1).item())
        psnr_val = compute_psnr(gen_t, gt_t)
        ssim_val = compute_ssim_simple(gen_t, gt_t)
        return lpips_val, psnr_val, ssim_val


# ─────────────────────── per-pairing per-subtask runner ───────────────────────

def collect_pairs(subset_path: str, pairing: str) -> List[Dict]:
    gt_field, gen_field = PAIRINGS[pairing]
    samples = json.load(open(subset_path))
    pairs = []
    for s in samples:
        gt = s.get(gt_field)
        gen = s.get(gen_field)
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
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dino_model", default="facebook/dinov2-base")
    parser.add_argument("--siglip_model", default="google/siglip-large-patch16-384",
                        help="SigLIP backbone for the primary (aspect-preserving) image-image cosine.")
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--metrics", nargs="*",
                        default=["siglip", "dino", "lpips", "psnr", "ssim"])
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for DINO/SigLIP")
    parser.add_argument("--num_samples", type=int, default=None, help="Cap per (pairing, subtask) for smoke")
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    pairings_to_run = list(PAIRINGS.keys()) if args.pairing == "all" else [args.pairing]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Lazy-load metrics
    siglip = SiglipSimilarity(args.siglip_model, args.device) if "siglip" in args.metrics else None
    dino = DinoSimilarity(args.dino_model, args.device) if "dino" in args.metrics else None
    pixel = PixelMetrics(args.device, args.lpips_net) if any(m in args.metrics for m in ("lpips","psnr","ssim")) else None

    for pairing in pairings_to_run:
        for subtask in args.subtasks:
            subset_path = Path(args.subset_dir) / f"{subtask}.json"
            if not subset_path.exists():
                print(f"[{pairing}/{subtask}] subset missing: {subset_path}")
                continue
            pairs = collect_pairs(str(subset_path), pairing)
            if args.num_samples is not None:
                pairs = pairs[: args.num_samples]
            if not pairs:
                print(f"[{pairing}/{subtask}] no usable pairs")
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
                    # Resume only rows that already have ALL requested metrics.
                    required_keys = []
                    if siglip is not None: required_keys.append("siglip_cos")
                    if dino is not None: required_keys.append("dino_cos")
                    if pixel is not None: required_keys += ["lpips", "psnr", "ssim"]
                    kept = []
                    for r in prev_results:
                        if all(k in r and not (isinstance(r[k], float) and math.isnan(r[k])) for k in required_keys):
                            kept.append(r)
                            done_ids.add(r["sample_id"])
                    prev_results = kept
                except Exception as e:
                    print(f"[{pairing}/{subtask}] resume failed: {e}")
            todo = [p for p in pairs if p["sample_id"] not in done_ids]
            print(f"[{pairing}/{subtask}] total={len(pairs)} resume={len(done_ids)} todo={len(todo)}")
            if not todo:
                continue

            t0 = time.time()
            results: List[Dict] = list(prev_results)

            # Batch DINO for speed; per-sample LPIPS to keep memory predictable.
            for i in tqdm(range(0, len(todo), args.batch_size), desc=f"{pairing}/{subtask}"):
                batch = todo[i : i + args.batch_size]
                row_metrics: List[Dict] = []
                # Load PILs once
                gt_pils, gen_pils = [], []
                for p in batch:
                    try:
                        gt_pils.append(Image.open(p["gt_image_path"]).convert("RGB"))
                        gen_pils.append(Image.open(p["generated_image_path"]).convert("RGB"))
                    except Exception as e:
                        gt_pils.append(None); gen_pils.append(None)
                        print(f"  failed to open {p['sample_id']}: {e}")

                # SigLIP batched (primary, aspect-preserving)
                siglip_scores: List[float] = [float("nan")] * len(batch)
                if siglip is not None:
                    valid_idx = [j for j, im in enumerate(gt_pils) if im is not None and gen_pils[j] is not None]
                    if valid_idx:
                        gt_valid = [gt_pils[j] for j in valid_idx]
                        gen_valid = [gen_pils[j] for j in valid_idx]
                        sims = siglip.cos_similarity(gt_valid, gen_valid)
                        for j, s in zip(valid_idx, sims):
                            siglip_scores[j] = s

                # DINO batched (secondary)
                dino_scores: List[float] = [float("nan")] * len(batch)
                if dino is not None:
                    valid_idx = [j for j, im in enumerate(gt_pils) if im is not None and gen_pils[j] is not None]
                    if valid_idx:
                        gt_valid = [gt_pils[j] for j in valid_idx]
                        gen_valid = [gen_pils[j] for j in valid_idx]
                        sims = dino.cos_similarity(gt_valid, gen_valid)
                        for j, s in zip(valid_idx, sims):
                            dino_scores[j] = s

                # LPIPS/PSNR/SSIM per-sample
                for j, p in enumerate(batch):
                    row = {
                        "sample_id": p["sample_id"],
                        "subtask": p["subtask"],
                        "scene_id": p["scene_id"],
                        "pairing": pairing,
                        "gt_image_path": p["gt_image_path"],
                        "generated_image_path": p["generated_image_path"],
                        "siglip_cos": siglip_scores[j],
                        "dino_cos": dino_scores[j],
                    }
                    if pixel is not None and gt_pils[j] is not None and gen_pils[j] is not None:
                        try:
                            l, ps, ss = pixel.compute(gt_pils[j], gen_pils[j])
                            row["lpips"] = l
                            row["psnr"] = ps
                            row["ssim"] = ss
                        except Exception as e:
                            print(f"  lpips/psnr/ssim failed on {p['sample_id']}: {e}")
                            row["lpips"] = float("nan")
                            row["psnr"] = float("nan")
                            row["ssim"] = float("nan")
                    results.append(row)

                # Periodic flush
                if (i // args.batch_size) % 4 == 0:
                    _flush(out_path, results, args, pairing, subtask)

            _flush(out_path, results, args, pairing, subtask)
            dt = time.time() - t0
            n_ok = sum(1 for r in results if not (isinstance(r.get("dino_cos"), float) and math.isnan(r["dino_cos"])))
            print(f"[{pairing}/{subtask}] wrote {len(results)} rows ({n_ok} ok) in {dt:.1f}s")


def _flush(out_path: Path, results: List[Dict], args, pairing: str, subtask: str):
    payload = {
        "config": {
            "pairing": pairing,
            "subtask": subtask,
            "siglip_model": args.siglip_model,
            "dino_model": args.dino_model,
            "lpips_net": args.lpips_net,
            "metrics": args.metrics,
        },
        "n_rows": len(results),
        "results": results,
    }
    tmp = str(out_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_path)


if __name__ == "__main__":
    main()
