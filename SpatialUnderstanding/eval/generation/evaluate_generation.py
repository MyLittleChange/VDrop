#!/usr/bin/env python3
"""
Evaluate spatial view generation results.

Computes:
  - Part A: Pixel metrics (PSNR, SSIM, LPIPS) between generated and GT images
  - Part B: Depth consistency (Scale-Invariant RMSE via DepthAnythingV2)
  - Part C: LLM instruction following (Gemini-based spatial claim verification)

Usage:
    python evaluate_generation.py --results_dir /path/to/results --metrics psnr ssim lpips depth llm
    python evaluate_generation.py --mode merge --results_dir /path/to/results
"""

import argparse
import glob
import json
import math
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ─────────────────────── optional imports ───────────────────────

_LPIPS_AVAILABLE = False
try:
    import lpips as lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    pass

_TORCHMETRICS_AVAILABLE = False
try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    _TORCHMETRICS_AVAILABLE = True
except ImportError:
    pass

_DEPTH_AVAILABLE = False
try:
    from transformers import pipeline as hf_pipeline
    _DEPTH_AVAILABLE = True
except ImportError:
    pass

_GEMINI_AVAILABLE = False
try:
    from google import genai
    from google.genai import types
    _GEMINI_AVAILABLE = True
except ImportError:
    pass


# ─────────────────────── constants ───────────────────────

TARGET_SIZE = (1024, 576)  # (width, height) matching generated images


# ─────────────────────── checkpoint manager ───────────────────────

class CheckpointManager:
    """Manages checkpointing for resumable LLM evaluation using sample_id as unique key."""

    def __init__(self, checkpoint_path: str, save_interval: int = 10):
        self.checkpoint_path = checkpoint_path
        self.save_interval = save_interval
        self.lock = threading.Lock()
        self.results = {}
        self.completed_count = 0

    def load_checkpoint(self) -> Dict[str, Dict]:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r") as f:
                    data = json.load(f)
                    all_results = data.get("results", [])
                    self.results = {r["sample_id"]: r for r in all_results}
                    print(f"Loaded checkpoint with {len(self.results)} completed samples")
                    return self.results
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load checkpoint: {e}")
        return {}

    def get_completed_sample_ids(self) -> set:
        return set(self.results.keys())

    def add_result(self, result: Dict):
        with self.lock:
            self.results[result["sample_id"]] = result
            self.completed_count += 1
            if self.completed_count % self.save_interval == 0:
                self._save_checkpoint()

    def _save_checkpoint(self):
        checkpoint_data = {
            "results": list(self.results.values()),
            "total_completed": len(self.results),
        }
        temp_path = self.checkpoint_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(checkpoint_data, f)
        os.replace(temp_path, self.checkpoint_path)

    def save_final(self):
        with self.lock:
            self._save_checkpoint()
            print(f"Checkpoint saved: {len(self.results)} samples")

    def get_all_results(self) -> List[Dict]:
        with self.lock:
            return list(self.results.values())


# ─────────────────────── sample loading ───────────────────────

def load_samples(results_dir: str, shard_pattern: str) -> List[Dict]:
    """Load and merge all shard result files, deduplicate by sample_id."""
    shard_files = sorted(glob.glob(os.path.join(results_dir, shard_pattern)))
    if not shard_files:
        raise FileNotFoundError(
            f"No files matching '{shard_pattern}' found in {results_dir}"
        )

    all_samples = []
    seen_ids = set()
    for sf in shard_files:
        with open(sf) as f:
            data = json.load(f)
        results = data.get("results", data) if isinstance(data, dict) else data
        if isinstance(results, dict):
            results = list(results.values())
        for r in results:
            sid = r.get("sample_id", "")
            if sid and sid not in seen_ids:
                # Verify generated image exists
                gen_path = r.get("generated_image_path", "")
                if gen_path and os.path.exists(gen_path):
                    seen_ids.add(sid)
                    all_samples.append(r)

    all_samples.sort(key=lambda x: x.get("sample_id", ""))
    return all_samples


def apply_sharding(samples: List[Dict], shard_spec: str) -> List[Dict]:
    """Apply 'index/total' shard slicing."""
    shard_idx, total_shards = map(int, shard_spec.split("/"))
    total = len(samples)
    shard_size = total // total_shards
    remainder = total % total_shards
    start = shard_idx * shard_size + min(shard_idx, remainder)
    end = start + shard_size + (1 if shard_idx < remainder else 0)
    return samples[start:end]


# ─────────────────────── image loading ───────────────────────

def load_image_pair(sample: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load generated and GT images, resize GT to match generated size.

    Returns tensors of shape (1, 3, H, W) in [0, 1].
    """
    gen_img = Image.open(sample["generated_image_path"]).convert("RGB")
    gt_img = Image.open(sample["gt_image_path"]).convert("RGB")
    gt_img = gt_img.resize(TARGET_SIZE, Image.BICUBIC)

    import torchvision.transforms.functional as TF
    gen_t = TF.to_tensor(gen_img).unsqueeze(0)
    gt_t = TF.to_tensor(gt_img).unsqueeze(0)
    return gen_t, gt_t


# ─────────────────────── Part A: pixel metrics ───────────────────────

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def _manual_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Simple SSIM implementation using means and variances over the full image."""
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    mu_pred = pred.mean(dim=[2, 3], keepdim=True)
    mu_target = target.mean(dim=[2, 3], keepdim=True)
    sigma_pred_sq = ((pred - mu_pred) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma_target_sq = ((target - mu_target) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma_cross = ((pred - mu_pred) * (target - mu_target)).mean(dim=[2, 3], keepdim=True)

    ssim_map = (
        (2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)
    ) / (
        (mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    )
    return ssim_map.mean().item()


def compute_pixel_metrics(
    samples: List[Dict],
    metrics_to_compute: Set[str],
    lpips_net: str = "alex",
    device: str = "cuda",
) -> Dict[str, List[float]]:
    """Compute PSNR, SSIM, and/or LPIPS for all samples."""
    results = {m: [] for m in metrics_to_compute}

    # Initialize models
    lpips_model = None
    if "lpips" in metrics_to_compute:
        if not _LPIPS_AVAILABLE:
            raise ImportError("lpips package required. Install with: pip install lpips")
        lpips_model = lpips_lib.LPIPS(net=lpips_net).to(device).eval()

    ssim_metric = None
    if "ssim" in metrics_to_compute and _TORCHMETRICS_AVAILABLE:
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    for sample in tqdm(samples, desc="Pixel metrics"):
        try:
            gen_t, gt_t = load_image_pair(sample)

            if "psnr" in metrics_to_compute:
                results["psnr"].append(compute_psnr(gen_t, gt_t))

            if "ssim" in metrics_to_compute:
                if ssim_metric is not None:
                    val = ssim_metric(gen_t.to(device), gt_t.to(device)).item()
                else:
                    val = _manual_ssim(gen_t, gt_t)
                results["ssim"].append(val)

            if "lpips" in metrics_to_compute:
                pred_scaled = gen_t * 2 - 1
                target_scaled = gt_t * 2 - 1
                with torch.no_grad():
                    val = lpips_model(
                        pred_scaled.to(device), target_scaled.to(device)
                    ).item()
                results["lpips"].append(val)

        except Exception as e:
            print(f"Error on {sample.get('sample_id', '?')}: {e}")
            for m in metrics_to_compute:
                results[m].append(float("nan"))

    return results


# ─────────────────────── Part B: depth consistency ───────────────────────

def compute_depth_metrics(
    samples: List[Dict],
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
    device: str = "cuda",
) -> List[float]:
    """Compute Scale-Invariant RMSE between depth maps of generated and GT images."""
    if not _DEPTH_AVAILABLE:
        raise ImportError(
            "transformers package required. Install with: pip install transformers"
        )

    depth_estimator = hf_pipeline(
        "depth-estimation",
        model=model_id,
        device=device,
        torch_dtype=torch.float32,
    )

    si_rmse_values = []
    for sample in tqdm(samples, desc="Depth metrics"):
        try:
            gen_img = Image.open(sample["generated_image_path"]).convert("RGB")
            gt_img = Image.open(sample["gt_image_path"]).convert("RGB")
            gt_img = gt_img.resize(TARGET_SIZE, Image.BICUBIC)

            gen_result = depth_estimator(gen_img)
            gt_result = depth_estimator(gt_img)

            # Extract depth as numpy arrays
            gen_depth = np.array(gen_result["depth"]).astype(np.float64)
            gt_depth = np.array(gt_result["depth"]).astype(np.float64)

            # Resize to common size if needed
            if gen_depth.shape != gt_depth.shape:
                from PIL import Image as PILImage
                h, w = min(gen_depth.shape[0], gt_depth.shape[0]), min(gen_depth.shape[1], gt_depth.shape[1])
                gen_depth_img = PILImage.fromarray(gen_depth).resize((w, h), PILImage.BICUBIC)
                gt_depth_img = PILImage.fromarray(gt_depth).resize((w, h), PILImage.BICUBIC)
                gen_depth = np.array(gen_depth_img, dtype=np.float64)
                gt_depth = np.array(gt_depth_img, dtype=np.float64)

            eps = 1e-6
            gen_depth = np.clip(gen_depth, eps, None)
            gt_depth = np.clip(gt_depth, eps, None)

            log_diff = np.log(gen_depth) - np.log(gt_depth)
            si_rmse = np.sqrt(np.mean((log_diff - np.mean(log_diff)) ** 2))
            si_rmse_values.append(float(si_rmse))

        except Exception as e:
            print(f"Error on {sample.get('sample_id', '?')}: {e}")
            si_rmse_values.append(float("nan"))

    return si_rmse_values


# ─────────────────────── Part C: LLM instruction following ───────────────────────

INSTRUCTION_EVAL_PROMPT = """You are evaluating whether a generated image correctly follows a spatial description instruction.

## Instruction that was given to the image generator:
{instruction}

## Task:
Look at the provided generated image and evaluate whether it follows the spatial relationships described in the instruction.

For each spatial claim in the instruction, determine if it is satisfied in the image. Spatial claims include statements about:
- Object positions (left, right, center, foreground, background)
- Object presence (whether mentioned objects appear)
- Relative spatial relationships between objects
- Viewpoint/facing direction

Respond in the following JSON format:
{{
  "claims": [
    {{"claim": "<the spatial claim text>", "satisfied": true/false, "reason": "<brief reason>"}},
    ...
  ],
  "overall_score": <float between 0.0 and 1.0 representing fraction of claims satisfied>,
  "overall_assessment": "<brief overall assessment>"
}}

Respond ONLY with the JSON, no other text."""


class InstructionFollowingEvaluator:
    """Evaluates generated images against spatial instructions using Gemini."""

    def __init__(self, model_name: str = "gemini-2.5-flash-preview-05-20"):
        if not _GEMINI_AVAILABLE:
            raise ImportError(
                "google-genai package required. Install with: pip install google-genai"
            )
        api_key = os.environ.get("VisualCoT_GEMINI")
        if not api_key:
            raise ValueError("VisualCoT_GEMINI environment variable is required")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        print(f"Initialized Gemini client with model: {model_name}")

    def evaluate_sample(self, sample: Dict) -> Dict:
        """Evaluate a single sample. Returns result dict with scores."""
        instruction = sample["instruction"]
        gen_image_path = sample["generated_image_path"]

        with open(gen_image_path, "rb") as f:
            image_bytes = f.read()

        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        prompt = INSTRUCTION_EVAL_PROMPT.format(instruction=instruction)

        config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=2048,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[image_part, prompt],
            config=config,
        )

        response_text = response.text.strip()

        # Parse JSON from response (handle markdown code blocks)
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first and last lines (``` markers)
            lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            response_text = "\n".join(lines)

        parsed = json.loads(response_text)
        return {
            "sample_id": sample["sample_id"],
            "claims": parsed.get("claims", []),
            "overall_score": float(parsed.get("overall_score", 0.0)),
            "overall_assessment": parsed.get("overall_assessment", ""),
        }


def run_llm_evaluation(
    samples: List[Dict],
    model_name: str,
    max_workers: int,
    checkpoint_interval: int,
    output_file: str,
    no_resume: bool,
) -> List[Dict]:
    """Run Gemini-based instruction following evaluation with resumability."""
    checkpoint_path = output_file.replace(".json", "_llm_checkpoint.json")
    checkpoint_mgr = CheckpointManager(checkpoint_path, save_interval=checkpoint_interval)

    completed_ids = set()
    if not no_resume and os.path.exists(checkpoint_path):
        checkpoint_mgr.load_checkpoint()
        completed_ids = checkpoint_mgr.get_completed_sample_ids()

    evaluator = InstructionFollowingEvaluator(model_name=model_name)

    samples_to_process = [s for s in samples if s["sample_id"] not in completed_ids]
    print(f"LLM evaluation: {len(samples_to_process)} to process, {len(completed_ids)} already done")

    NUM_RETRIES = 3

    def process_one(sample: Dict) -> Optional[Dict]:
        for retry in range(NUM_RETRIES):
            try:
                return evaluator.evaluate_sample(sample)
            except Exception as e:
                if retry < NUM_RETRIES - 1:
                    time.sleep(2 ** retry)
                else:
                    print(f"Error on {sample['sample_id']} after {NUM_RETRIES} retries: {e}")
                    return {
                        "sample_id": sample["sample_id"],
                        "overall_score": 0.0,
                        "claims": [],
                        "overall_assessment": "",
                        "error": str(e),
                    }
        return None

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_one, s): s["sample_id"]
                for s in samples_to_process
            }
            with tqdm(total=len(futures), desc="LLM evaluation") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        checkpoint_mgr.add_result(result)
                    pbar.update(1)
    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        checkpoint_mgr.save_final()
        print("Checkpoint saved. Will auto-resume on next run.")
        raise

    checkpoint_mgr.save_final()
    return checkpoint_mgr.get_all_results()


# ─────────────────────── aggregation ───────────────────────

def aggregate_metrics(per_sample: Dict[str, List[float]]) -> Dict:
    """Compute mean, std, median, min, max for each metric."""
    agg = {}
    for name, values in per_sample.items():
        arr = np.array([v for v in values if not (isinstance(v, float) and math.isnan(v))])
        if len(arr) == 0:
            agg[name] = {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        else:
            agg[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": len(arr),
            }
    return agg


def assemble_per_sample_results(
    samples: List[Dict],
    per_sample_metrics: Dict[str, List[float]],
    llm_results: List[Dict],
) -> List[Dict]:
    """Merge pixel/depth metrics with LLM results."""
    llm_by_id = {r["sample_id"]: r for r in llm_results}

    results = []
    for i, sample in enumerate(samples):
        entry = {
            "sample_id": sample["sample_id"],
            "generated_image_path": sample.get("generated_image_path", ""),
            "gt_image_path": sample.get("gt_image_path", ""),
        }
        for metric_name, values in per_sample_metrics.items():
            if i < len(values):
                entry[metric_name] = values[i]

        llm_data = llm_by_id.get(sample["sample_id"], {})
        if llm_data:
            entry["llm_instruction_score"] = llm_data.get("overall_score", 0.0)
            entry["llm_claims"] = llm_data.get("claims", [])
            entry["llm_assessment"] = llm_data.get("overall_assessment", "")
            if "error" in llm_data:
                entry["llm_error"] = llm_data["error"]

        results.append(entry)
    return results


# ─────────────────────── merge mode ───────────────────────

def merge_evaluation_results(results_dir: str, output_file: str):
    """Merge evaluation_results_shard*.json files into a single output."""
    shard_files = sorted(glob.glob(os.path.join(results_dir, "evaluation_results_shard*.json")))
    if not shard_files:
        print(f"No evaluation_results_shard*.json files found in {results_dir}")
        return

    print(f"Merging {len(shard_files)} shard files...")

    all_results = []
    seen_ids = set()
    merged_config = None

    for sf in shard_files:
        with open(sf) as f:
            data = json.load(f)
        if merged_config is None:
            merged_config = data.get("config", {})
        for r in data.get("results", []):
            sid = r.get("sample_id", "")
            if sid and sid not in seen_ids:
                seen_ids.add(sid)
                all_results.append(r)

    all_results.sort(key=lambda x: x.get("sample_id", ""))

    # Recompute aggregates
    metric_names = [
        k for k in all_results[0]
        if isinstance(all_results[0].get(k), (int, float)) and k != "llm_instruction_score"
    ]
    if any("llm_instruction_score" in r for r in all_results):
        metric_names.append("llm_instruction_score")

    per_sample = {}
    for m in metric_names:
        per_sample[m] = [r.get(m, float("nan")) for r in all_results]
    agg = aggregate_metrics(per_sample)

    merged_config["num_samples"] = len(all_results)
    merged_config["shard"] = None
    merged_config["merged_from"] = [os.path.basename(f) for f in shard_files]

    output_data = {
        "config": merged_config,
        "metrics": agg,
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Merged {len(all_results)} samples -> {output_file}")
    print_summary(agg)


# ─────────────────────── summary printer ───────────────────────

def print_summary(agg: Dict):
    print(f"\n{'=' * 60}")
    print("Evaluation Summary")
    print(f"{'=' * 60}")
    for name, stats in agg.items():
        mean = stats.get("mean", 0)
        std = stats.get("std", 0)
        count = stats.get("count", 0)
        print(f"  {name:>25s}: {mean:.4f} ± {std:.4f}  (n={count})")
    print(f"{'=' * 60}")


# ─────────────────────── main ───────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate spatial view generation results"
    )

    parser.add_argument(
        "--results_dir",
        default="/path/to/scratch/VisualCoT/BAGEL_description_orbit_anchor",
        help="Directory with generated results (contains results_summary_shard*.json)",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output JSON path. Default: {results_dir}/evaluation_results.json",
    )
    parser.add_argument(
        "--shard_pattern",
        default="results_summary_shard*.json",
        help="Glob pattern for result summary files",
    )

    # Mode
    parser.add_argument(
        "--mode",
        default="evaluate",
        choices=["evaluate", "merge"],
        help="evaluate: run metrics; merge: combine sharded results",
    )

    # Metric selection
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "ssim", "lpips", "depth", "llm"],
        choices=["psnr", "ssim", "lpips", "depth", "llm"],
        help="Which metrics to compute",
    )

    # Sharding
    parser.add_argument(
        "--shard", type=str, default=None, help="Shard spec 'index/total'"
    )

    # LLM config
    parser.add_argument("--gemini_model", default="gemini-2.5-flash-preview-05-20")
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--no_resume", action="store_true")

    # Depth config
    parser.add_argument(
        "--depth_model",
        default="depth-anything/Depth-Anything-V2-Small-hf",
    )

    # LPIPS config
    parser.add_argument(
        "--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"]
    )

    # General
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine output path
    if args.output_file:
        output_file = args.output_file
    elif args.shard:
        shard_idx = args.shard.split("/")[0]
        output_file = os.path.join(args.results_dir, f"evaluation_results_shard{shard_idx}.json")
    else:
        output_file = os.path.join(args.results_dir, "evaluation_results.json")

    # Merge mode
    if args.mode == "merge":
        merge_output = args.output_file or os.path.join(
            args.results_dir, "evaluation_results_merged.json"
        )
        merge_evaluation_results(args.results_dir, merge_output)
        return

    # Load samples
    samples = load_samples(args.results_dir, args.shard_pattern)
    print(f"Loaded {len(samples)} samples from {args.results_dir}")

    # Apply sharding
    if args.shard:
        samples = apply_sharding(samples, args.shard)
        print(f"Shard {args.shard}: processing {len(samples)} samples")

    # Apply num_samples limit
    if args.num_samples and args.num_samples < len(samples):
        samples = samples[: args.num_samples]
        print(f"Limited to {len(samples)} samples")

    if not samples:
        print("No samples to evaluate.")
        return

    metrics_requested = set(args.metrics)
    per_sample_metrics = {}

    # Part A: Pixel metrics
    pixel_metrics = metrics_requested & {"psnr", "ssim", "lpips"}
    if pixel_metrics:
        print(f"\n--- Part A: Pixel Metrics ({', '.join(sorted(pixel_metrics))}) ---")
        pixel_results = compute_pixel_metrics(
            samples, pixel_metrics, args.lpips_net, args.device
        )
        per_sample_metrics.update(pixel_results)

    # Part B: Depth
    if "depth" in metrics_requested:
        print("\n--- Part B: Depth Consistency ---")
        depth_values = compute_depth_metrics(samples, args.depth_model, args.device)
        per_sample_metrics["depth_si_rmse"] = depth_values

    # Part C: LLM evaluation
    llm_results = []
    if "llm" in metrics_requested:
        print("\n--- Part C: LLM Instruction Following ---")
        llm_results = run_llm_evaluation(
            samples,
            args.gemini_model,
            args.max_workers,
            args.checkpoint_interval,
            output_file,
            args.no_resume,
        )

    # Assemble per-sample results
    results_list = assemble_per_sample_results(samples, per_sample_metrics, llm_results)

    # Build aggregate metrics dict
    agg_input = dict(per_sample_metrics)
    if llm_results:
        agg_input["llm_instruction_score"] = [
            r.get("overall_score", 0.0) for r in llm_results
        ]
    agg = aggregate_metrics(agg_input)

    # Save
    config = {
        "results_dir": args.results_dir,
        "metrics_computed": args.metrics,
        "num_samples": len(samples),
        "shard": args.shard,
        "lpips_net": args.lpips_net if "lpips" in metrics_requested else None,
        "depth_model": args.depth_model if "depth" in metrics_requested else None,
        "gemini_model": args.gemini_model if "llm" in metrics_requested else None,
        "device": args.device,
    }

    output_data = {
        "config": config,
        "metrics": agg,
        "results": results_list,
    }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print_summary(agg)


if __name__ == "__main__":
    main()
