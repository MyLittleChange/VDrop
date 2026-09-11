"""Merge a LoRA adapter checkpoint into a base BAGEL model.

Produces a full model.safetensors (same format as BAGEL_format_* eval dirs)
by applying: W_merged = W_base + (lora_B @ lora_A) * (alpha / rank)

Usage:
    python tools/merge_lora.py \
        --base_model_path /path/to/BAGEL-7B-MoT \
        --lora_adapter_path /path/to/lora_checkpoint_dir \
        --output_path /path/to/output_dir
"""
import argparse
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_safetensors(path: str) -> dict:
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base BAGEL model")
    parser.add_argument("--base_model_path", default='/path/to/scratch/models/BAGEL-7B-MoT',
                        help="Base model dir: HF format (has ema.safetensors) "
                             "or flat format (has model.safetensors)")
    parser.add_argument("--lora_adapter_path", default='/path/to/scratch/VisualCoT/BAGEL_checkpoints/training_data_mix_balance_matterport_point_matching_no_think_lora/0009000',
                        help="Path to lora_adapter.safetensors or a dir containing it")
    parser.add_argument("--output_path", default='/path/to/scratch/VisualCoT/BAGEL_checkpoints/training_data_mix_balance_matterport_point_matching_no_think_lora',
                        help="Output directory for merged checkpoint")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=64.0)
    args = parser.parse_args()

    # Resolve LoRA adapter path
    lora_path = args.lora_adapter_path
    if os.path.isdir(lora_path):
        lora_path = os.path.join(lora_path, "lora_adapter.safetensors")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA adapter not found: {lora_path}")

    # Detect base weights file: prefer model.safetensors, fall back to ema.safetensors
    base_model_sf = os.path.join(args.base_model_path, "model.safetensors")
    base_ema_sf = os.path.join(args.base_model_path, "ema.safetensors")
    if os.path.exists(base_model_sf):
        base_sf_path = base_model_sf
        print(f"Loading base from model.safetensors: {base_model_sf}")
    elif os.path.exists(base_ema_sf):
        base_sf_path = base_ema_sf
        print(f"Loading base from ema.safetensors: {base_ema_sf}")
    else:
        raise FileNotFoundError(
            f"No model.safetensors or ema.safetensors found in {args.base_model_path}"
        )

    print("Loading base weights...")
    base_weights = load_safetensors(base_sf_path)
    print(f"  {len(base_weights)} base tensors")

    print("Loading LoRA adapter...")
    lora_weights = load_safetensors(lora_path)
    print(f"  {len(lora_weights)} adapter tensors")

    scale = args.lora_alpha / args.lora_rank
    print(f"  rank={args.lora_rank}, alpha={args.lora_alpha}, scale={scale:.4f}")

    # Start from a copy of base weights (values replaced, not mutated)
    merged = dict(base_weights)

    # Apply LoRA deltas
    lora_A_keys = sorted(k for k in lora_weights if ".lora_A." in k)
    merged_count = 0
    missing_base_keys = []

    for lora_A_key in lora_A_keys:
        lora_B_key = lora_A_key.replace(".lora_A.", ".lora_B.")
        base_key = re.sub(r"\.lora_A\.default\.weight$", ".weight", lora_A_key)

        if base_key not in merged:
            missing_base_keys.append(base_key)
            continue

        lora_A = lora_weights[lora_A_key].float()   # (rank, in_features)
        lora_B = lora_weights[lora_B_key].float()   # (out_features, rank)
        base_w = merged[base_key].float()

        delta = (lora_B @ lora_A) * scale
        merged[base_key] = (base_w + delta).to(torch.bfloat16)
        merged_count += 1

    if missing_base_keys:
        print(f"WARNING: {len(missing_base_keys)} LoRA targets not found in base:")
        for k in missing_base_keys[:5]:
            print(f"  {k}")

    print(f"Applied {merged_count} LoRA deltas")

    # Replace embed_tokens and lm_head with fully-trained adapter versions
    full_rank_keys = [
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
    ]
    for key in full_rank_keys:
        if key in lora_weights:
            merged[key] = lora_weights[key].to(torch.bfloat16)
            print(f"Replaced {key} with adapter version")

    # Ensure all remaining weights are bfloat16
    for k in merged:
        if merged[k].dtype != torch.bfloat16:
            merged[k] = merged[k].to(torch.bfloat16)

    # Save merged model
    os.makedirs(args.output_path, exist_ok=True)
    out_model_path = os.path.join(args.output_path, "model.safetensors")
    print(f"Saving merged model to {out_model_path} ...")
    save_file(merged, out_model_path)
    print(f"  Saved {len(merged)} tensors")

    # Copy support files from base model dir
    support_files = [
        "ae.safetensors",
        "config.json",
        "generation_config.json",
        "llm_config.json",
        "merges.txt",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vit_config.json",
        "vocab.json",
    ]
    copied = []
    for fname in support_files:
        src = os.path.join(args.base_model_path, fname)
        dst = os.path.join(args.output_path, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied.append(fname)
    print(f"Copied {len(copied)} support files: {', '.join(copied)}")
    print("Done.")


if __name__ == "__main__":
    main()
