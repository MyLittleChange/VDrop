# ThinkMorph — BAGEL Cross-View Spatial Reasoning

Code release for our study of **visual chain-of-thought (CoT) for cross-view indoor spatial
reasoning** on top of **[BAGEL-7B-MoT](https://github.com/ByteDance-Seed/Bagel)**, a unified
vision–language model that can *generate* images as part of its reasoning.

Given two camera views of the same indoor scene, the model answers multiple-choice questions
about spatial relationships (anchor, counting, relative-distance, relative-direction, mapping,
point-matching). We fine-tune BAGEL under four *thinking modes* and study whether the
model's self-generated intermediate image (the **bridge**) is actually *used* when it produces
the answer.

- 📄 Paper / website: [ThinkMorph](https://thinkmorph.github.io/) · [arXiv:2510.27492](https://arxiv.org/abs/2510.27492)
- 🤗 Model: [ThinkMorph/ThinkMorph-7B](https://huggingface.co/ThinkMorph/ThinkMorph-7B) · Datasets: [ThinkMorph](https://huggingface.co/ThinkMorph)

> This folder is a **curated, standalone extract** of the BAGEL-related code from our research
> repository. It bundles the BAGEL model core plus everything needed to build training data,
> train, run inference, and evaluate on all benchmarks reported in the paper. Data *rendering*
> (Blender/Infinigen) is documented but not shipped — see [Data Creation](#5-data-creation).

---

## Table of Contents
1. [The four thinking modes](#1-the-four-thinking-modes)
2. [Headline results](#2-headline-results)
3. [Repository layout](#3-repository-layout)
4. [Installation](#4-installation)
5. [Data creation](#5-data-creation)
6. [Training](#6-training)
7. [Inference & evaluation](#7-inference--evaluation)
8. [Bridge-necessity analysis (the core research question)](#8-bridge-necessity-analysis)
9. [Adapting paths to your environment](#9-adapting-paths-to-your-environment)
10. [Citation & acknowledgements](#10-citation--acknowledgements)

---

## 1. The four thinking modes

The model is trained with the same two input views `[V1, V2]` under four supervision formats:

| Mode | Output format | What the model does | File format |
| --- | --- | --- | --- |
| `no_think` | `<answer>X</answer>` | Answer directly from the two views | JSONL (ShareGPT) |
| `text_think` | `<think>…</think><answer>X</answer>` | Textual reasoning, then answer | JSONL (ShareGPT) |
| `visual_only` | `<image_start>…<image_end><answer>X</answer>` | **Generate a bridge image**, then answer | Parquet (Arrow) |
| `interleaved` | `<think>…</think><image_start>…<image_end><answer>X</answer>` | Text + generated image, then answer | Parquet (Arrow) |

The **bridge image** the model is trained to generate can be one of several cross-view
representations, each a separate training recipe studied in the paper:

- **Panorama** — a 360° panoramic view of the scene.
- **Top-down / map** — a bird's-eye orthographic layout (matplotlib BEV or photoreal Blender).
- **Corner-view** — a photoreal ultra-wide "room overview" (real-estate-listing composition).
- **Point-matching** — a side-by-side annotation marking the same object in both views.

### Bridge-necessity training
A central finding is that the `visual_only` model tends to **generate a bridge to satisfy the
image objective but then ignore it**, reading the answer straight from `[V1, V2]` (a shortcut).
We add **partial-view attention masking** during training that hides part of one input view from
the answer's attention so the *only* path to cross-view information is *through the bridge*.
See [`docs/`](docs/) and [`SpatialUnderstanding/BRIDGE_MASKED_TRAINING_README.md`](SpatialUnderstanding/BRIDGE_MASKED_TRAINING_README.md).

---

## 2. Headline results

Accuracy (%) on **Setting A** (trained on Anchor / Counting / Rel-Dist / Rel-Dir; Map held out
as OOD). LoRA SFT, r=32, α=64, 7K balanced samples. Full tables and all ablations are in the
[paper](https://arxiv.org/abs/2510.27492).

| Method | Anchor | Count. | Rel-Dist | Rel-Dir | MMSI | MindCube | STARE | BLINK |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BAGEL (vanilla) | 18.0 | 42.1 | 24.4 | 21.2 | 26.9 | 31.7 | 28.0 | 45.1 |
| `no_think` (SFT) | 86.8 | 82.4 | 67.6 | 85.6 | 27.4 | 41.1 | 24.8 | 45.1 |
| Panorama `visual_only` (SFT) | 93.6 | 83.6 | 76.4 | 87.2 | 24.9 | 36.9 | 32.4 | 55.6 |
| Panorama + bridge-masking | 89.2 | 78.8 | 74.8 | 93.2 | 26.0 | 34.1 | 35.6 | **62.4** |

**Read:** SFT lifts every in-domain task by 40–70 points over vanilla BAGEL. Generating a bridge
(panorama) further helps the held-out real-world benchmarks (BLINK, STARE). Bridge-necessity
masking pushes the OOD generalization (BLINK/STARE) the furthest — see §8 for the causal probes
that test whether the bridge is genuinely load-bearing.

---

## 3. Repository layout

```
ThinkMorph-BAGEL-release/
├── modeling/                     # BAGEL model core (BAGEL, Qwen2-NaViT, SigLIP, VAE) — from upstream Bagel
├── data/                         # Dataset loading, transforms, tokenization, packing
│   ├── interleave_datasets/      #   Interleaved text+image iterable datasets
│   ├── bridge_masking_utils.py   #   ★ Partial-view attention-mask builder + curriculum
│   ├── bridge_masked_packed_dataset.py   # ★ View-role-aware sequence packer
│   └── dataset_info.py           #   Dataset registry (wire new SFT parquet dirs here)
├── train/
│   ├── pretrain_unified_navit.py             # Main FSDP training loop
│   └── pretrain_unified_navit_bridge_masked.py  # ★ Bridge-masking shim (monkey-patches packer)
├── configs/
│   ├── data/                     # Dataset YAML configs (spatial_reasoning*, visual_only_mix_all, …)
│   └── train/                    # Reference SLURM training scripts
│
├── inferencer.py                 # Interleaved text+image inference engine (+ inference-time bridge blinding)
├── inference/                    # Per-benchmark inference drivers
│   ├── run_inference_bagel_*.py  #   spatial, mindcube, omni, stare, blink, vsibench, …
│   └── eval_mmsi_bench.py        #   MMSI-Bench judge (Gemini / Qwen-VL)
├── scripts/                      # SLURM eval submit + LoRA-merge-then-submit wrappers
│
├── SpatialUnderstanding/
│   ├── data_creation/training_data/   # ★ SFT-assembly: create_*_sft_data.py mixers + scene indexers
│   ├── eval/                          # Per-benchmark launchers + merge/scoring scripts
│   │   ├── spatial/  mmsi/  mindcube/  omnispatial/  stare/  blink/  generation/  vsibench/
│   │   └── bagel/    #   ThinkMorph-7B and Bagel-Zebra-CoT baseline drivers
│   ├── spatial_*.py                   # Spatial dataset / trainer / inferencer helpers
│   ├── train_spatial_reasoning_tamia.sh
│   ├── train_spatial_reasoning_bridge_masked_tamia.sh
│   └── BRIDGE_MASKED_TRAINING_README.md
│
├── tools/
│   ├── merge_lora.py                  # Merge a LoRA adapter into the base BAGEL weights
│   ├── answer_attention_probe*.py     # ★ Answer-token attention probe (does the answer attend to the bridge?)
│   └── …                              # metric extraction, manifests, plotting
├── tests/                             # CPU unit tests for the masking + probes (no pytest required)
├── docs/                             # AttentionProbReadMe.md, TRAIN.md, two_reader_informativeness_design.md
│
├── cluster.env.example               # Cluster path/config template → copy to cluster.env
├── setup_env.sh                      # Env bootstrap (uv venv + requirements)
└── requirements.txt
```

★ = code specific to this work (everything else under `modeling/`, and much of `data/`+`train/`,
is inherited from upstream BAGEL and lightly modified).

---

## 4. Installation

Requires a CUDA GPU (training uses 4×A100-80GB with FSDP; single-GPU inference is supported).

```bash
# 1. Configure your cluster paths (venv location, scratch, model dir, …)
cp cluster.env.example cluster.env
$EDITOR cluster.env

# 2. Create the environment (uv venv + requirements.txt + CUDA-matched torch)
bash setup_env.sh
source <VENV_PATH>/bin/activate        # the VENV_PATH you set in cluster.env
```

Core dependencies (see [`requirements.txt`](requirements.txt)): `transformers==4.49.0`,
`accelerate`, `pyarrow==11.0.0`, `safetensors`, `einops`, `flash-attn`/`triton`,
`decord`, `opencv-python`. `torch`/`torchvision` are installed separately with a CUDA index URL
(handled by `setup_env.sh`) to avoid CPU-only wheels.

**Base model:** download [BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) and
point `BAGEL_MODEL_PATH` (in `cluster.env`) at it.

---

## 5. Data creation

Training data is produced in three stages. **This release ships Stage 3 (SFT assembly).**
Stages 1–2 (Blender/Infinigen rendering + Gemini scene annotation) are cluster- and
Blender-specific and are *documented here* rather than shipped; the rendered scenes and the
released SFT parquet/JSONL are on 🤗 [ThinkMorph](https://huggingface.co/ThinkMorph).

| Stage | What | Where |
| --- | --- | --- |
| 1. Rendering | Multi-view / panorama / top-down / corner-view renders from Infinigen & Matterport scenes (Blender) | *documented — not shipped* |
| 2. Annotation | Visible-object detection (Qwen-VL) + scene descriptions + QA + text-CoT (Gemini) | *documented — not shipped* |
| 3. **SFT assembly** | Turn annotated scenes into training Parquet/JSONL for the 4 thinking modes | [`SpatialUnderstanding/data_creation/training_data/`](SpatialUnderstanding/data_creation/training_data/) |

**Stage-3 mixers** (each takes annotated scenes → per-mode training files):

```bash
cd SpatialUnderstanding/data_creation/training_data

# Main Infinigen mixer (anchor/counting/dist/dir/perspective × 4 thinking modes)
python create_mix_all_sft_data.py --thinking_mode visual_only   # or no_thinking / text_thinking / interleaved

# Bridge variants
python create_mix_topdown_sft_data_round3.py     # photoreal top-down bridge
python create_mix_corner_view_sft_data.py        # photoreal corner-view bridge
python create_mix_pm_sft_data.py                 # point-matching bridge
python create_matterport_point_matching_sft_data.py
python create_matterport_rotation_sft_data.py
python create_omnispatial_sft_data.py            # real-world supplementary SFT
```

Wire a new parquet directory into training by adding it to
[`data/dataset_info.py`](data/dataset_info.py) under `spatial_reasoning` / `visual_only_thinking`.
The full data-pipeline design (perturbed-negatives for the map task, corner-view camera geometry,
point-matching annotation, coverage tables) is documented in the [paper](https://arxiv.org/abs/2510.27492).

---

## 6. Training

Entry point: `train/pretrain_unified_navit.py`, launched via a SLURM/torchrun wrapper.
Base hyperparameters: `lr=1e-5`, 4×A100-80GB (FSDP), `save_every=50`. LoRA (where used): `r=32, α=64`.

```bash
# Standard SFT (edit paths + dataset config inside the script first)
sbatch SpatialUnderstanding/train_spatial_reasoning_tamia.sh
#   → torchrun … train/pretrain_unified_navit.py \
#        --model_path $BAGEL_MODEL_PATH \
#        --dataset_config_file ./data/configs/spatial_reasoning.yaml \
#        --layer_module Qwen2MoTDecoderLayer --finetune_from_hf True …

# Bridge-necessity (partial-view masking) SFT
sbatch SpatialUnderstanding/train_spatial_reasoning_bridge_masked_tamia.sh
```

The bridge-masked launcher is a thin wrapper that runs
`train/pretrain_unified_navit_bridge_masked.py` (which monkey-patches the packed dataset — the
**original trainer is untouched**) and exports `BRIDGE_MASK_*` env vars:

| Env var | Default | Meaning |
| --- | --- | --- |
| `BRIDGE_MASK_WARMUP_STEPS` | `500` | Full-attention warmup before any masking |
| `BRIDGE_MASK_ANNEAL_STEPS` | `1500` | Linear ramp of mask probability 0→1 |
| `BRIDGE_MASK_DROP_FRACTION` | `0.5` | Fraction of the chosen view's patches to hide |
| `BRIDGE_MASK_DROP_STRATEGY` | `region` | `region` (contiguous) or `random_patches` |
| `BRIDGE_MASK_DROP` | `random` | `random` (one of V1/V2), `both`, `V1`, `V2`, `none` |

After a LoRA run finishes, merge the adapter into base weights for evaluation:

```bash
python tools/merge_lora.py \
    --base_model_path  $BAGEL_MODEL_PATH \
    --lora_adapter_path <CKPT_DIR>/<STEP> \
    --output_path       <CKPT_DIR>/BAGEL_format_<name> \
    --lora_rank 32 --lora_alpha 64.0
```

---

## 7. Inference & evaluation

Inference is sharded across GPUs; each shard writes one JSON. **Never average the per-shard
numbers printed in the logs** — always run the benchmark's merge script, which deduplicates by
`sample_id` and reports the canonical `Overall Accuracy`.

### Generic spatial launcher

```bash
# 4-GPU sharded inference for one task (anchor|counting|distance|direction)
MODEL_PATH=<…/BAGEL_format_your_ckpt> \
OUTPUT_DIR=<…/results_dir> \
TASK=anchor THINK=false THINKING_MODE=no_thinking \
bash SpatialUnderstanding/eval/spatial/run_bagel_spatial_generic.sh

# Merge shards → canonical accuracy
python SpatialUnderstanding/eval/spatial/merge_bagel_spatial_results.py \
    --input_dir $OUTPUT_DIR --output_file $OUTPUT_DIR/inference_results_bagel_merged.json
```

> **`--image_shapes` rule (important):** it must match how the bridge was built at training time.
> Panorama / point-matching bridges (side-by-side ~2560×720) → `IMAGE_SHAPES="720 1024"`.
> All other checkpoints (top-down, corner-view, no-bridge, vanilla) → `IMAGE_SHAPES="720 720"` (the default).
> Wrong shape silently stretches every bridge image and tanks accuracy.

### Benchmark suite

| Benchmark | Inference driver | Merge / score |
| --- | --- | --- |
| COSMIC spatial (5 tasks) | `inference/run_inference_bagel_spatial.py` | `eval/spatial/merge_bagel_spatial_results.py` |
| BLINK-MultiView | `inference/run_inference_bagel_blink_multiview.py` | `eval/blink/merge_blink_results.py` |
| MindCube | `inference/run_inference_bagel_mindcube.py` | `eval/mindcube/merge_mindcube_results.py` |
| OmniSpatial | `inference/run_inference_bagel_omnispatial.py` | `eval/omnispatial/merge_and_eval_omnispatial.py` |
| STARE-Perspective | `inference/run_inference_bagel_stare_perspective.py` | `eval/stare/merge_and_eval_stare.py` |
| MMSI-Bench (judge-based) | `inference/eval_mmsi_bench.py` | `eval/mmsi/merge_and_eval_mmsi.sh` |
| VSI-Bench | `inference/run_inference_bagel_vsibench.py` | `eval/vsibench/…` |

Per-checkpoint, per-task launchers live under `SpatialUnderstanding/eval/<benchmark>/`. The
naming convention is `run_bagel_<benchmark>_<Ngpu>_<checkpoint-tag>.sh`. The many near-duplicate
scripts are the exact experiment matrix from the paper; for new checkpoints, copy the nearest
one and change `MODEL_PATH` / `OUTPUT_DIR`.

**MMSI-Bench** uses an LLM judge and needs an API key:
`export VisualCoT_GEMINI=<your-gemini-key>` (or pass `--api_key`). No keys are stored in this repo.

---

## 8. Bridge-necessity analysis

The research question — *does the model actually use its self-generated bridge to answer?* — is
attacked three ways. All are additive (flag-off ≡ upstream behavior).

| Probe | What it measures | Entry points | Doc |
| --- | --- | --- | --- |
| **Training-time masking** (§6) | Force the bridge to be the only cross-view path | `data/bridge_masking_utils.py`, `train/pretrain_unified_navit_bridge_masked.py` | [`BRIDGE_MASKED_TRAINING_README.md`](SpatialUnderstanding/BRIDGE_MASKED_TRAINING_README.md) |
| **Answer-attention probe** | *Whether* answer-token queries attend to the bridge (`V1`/`V2`/`bridge_vae`/`bridge_vit`/text mass) | `tools/answer_attention_probe.py`, `tools/answer_attention_probe_batch.py` | [`docs/AttentionProbReadMe.md`](docs/AttentionProbReadMe.md) |
| **Inference-time blinding** ("generate-then-blind") | *Causal* test: zero the bridge KV-cache slice before decoding the answer and measure the accuracy drop | `inferencer.py` (`--inference_bridge_mask`), `inference/run_inference_bagel_spatial.py` | see paper |

```bash
# CPU unit tests for all three (no pytest needed)
python tests/test_bridge_partial_view_mask.py     # → 8/8 passed
python tests/test_inference_bridge_mask.py         # → 6/6 passed
python tests/test_answer_attention_probe.py

# Inference-time blinding: add the flag to any spatial launcher
#   inference/run_inference_bagel_spatial.py … --inference_bridge_mask
```

**Finding:** for the SFT'd checkpoints the accuracy drop
from blinding the bridge is within ±2 pt on most tasks — i.e. the bridge is *not* causally
necessary at inference, confirming the shortcut hypothesis even for the bridge-masked-trained model.

---

## 9. Adapting paths to your environment

All author-specific cluster paths have been replaced with **placeholder prefixes** so nothing
personal ships in the code. Before running, retarget these four placeholders to your environment:

| Placeholder | Meaning | Set to |
| --- | --- | --- |
| `/path/to/ThinkMorph-BAGEL-release` | Where you cloned this release | the repo root |
| `/path/to/scratch` | Fast scratch: data, models, checkpoints, outputs | your scratch filesystem |
| `/path/to/home` | Home dir (venv, conda, tooling) | your home |
| `/path/to/archive` | Cold archive (raw `.tar.xz` scenes, rendering only) | your archive mount |

The canonical knobs live in [`cluster.env`](cluster.env.example) (`SCRIPT_DIR`, `SCRATCH_ROOT`,
`MODEL_ROOT`/`BAGEL_MODEL_PATH`, `DATA_ROOT`/`OUTPUT_ROOT`/`CKPT_ROOT`, `VENV_PATH`, `CUDA_MODULE`,
`PARTITION`, `GPU_TYPE`). Many individual `.sh` scripts still hardcode the placeholder paths
directly rather than sourcing `cluster.env`, so a one-shot global retarget is the fastest fix:

```bash
cp cluster.env.example cluster.env        # then edit it

# Retarget every placeholder in one pass (run from the repo root):
grep -rl '/path/to/ThinkMorph-BAGEL-release' . | xargs -r sed -i "s#/path/to/ThinkMorph-BAGEL-release#$PWD#g"
grep -rl '/path/to/scratch' .                  | xargs -r sed -i 's#/path/to/scratch#'"$YOUR_SCRATCH"'#g'
grep -rl '/path/to/home' .                     | xargs -r sed -i 's#/path/to/home#'"$HOME"'#g'
```

Also update the `#SBATCH --output=/…` / `--error=/…` directives, `source …/activate`, and
`module load cuda/…` lines to match your cluster (SBATCH directives are **not** shell-expanded,
so they can't reference variables — edit them to literal paths). Monitoring commands already use
`$USER`.

---

## 10. Citation & acknowledgements

This work builds directly on **[BAGEL-7B-MoT](https://github.com/ByteDance-Seed/Bagel)** by
ByteDance-Seed (Apache-2.0). We modify the training code to support our interleaved
text+image data format and the bridge-necessity masking; the model architecture under
`modeling/` is inherited from BAGEL. See BAGEL's [TRAIN.md](https://github.com/ByteDance-Seed/Bagel/blob/main/TRAIN.md).

```bibtex
@article{gu2025thinkmorph,
  title={ThinkMorph: Emergent Properties in Multimodal Interleaved Chain-of-Thought Reasoning},
  author={Gu, Jiawei and Hao, Yunzhuo and Wang, Huichen Will and Li, Linjie and Shieh, Michael Qizhe and Choi, Yejin and Krishna, Ranjay and Cheng, Yu},
  journal={arXiv preprint arXiv:2510.27492},
  year={2025}
}
```

See [`LICENSE`](LICENSE) for licensing terms.
