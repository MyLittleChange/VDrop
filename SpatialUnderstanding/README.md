# Spatial Understanding Pipeline

End-to-end pipeline for spatial view generation and spatial reasoning evaluation: render multi-view images from Infinigen scenes, annotate with Gemini, create training data, train models, and evaluate.

## Repository Structure

```
ThinkMorph/
├── modeling/                         # Model architectures (BAGEL, Qwen2, SigLIP, VAE)
├── train/                            # Training loop (FSDP, EMA, multi-thinking-mode)
│   └── pretrain_unified_navit.py
├── data/                             # Dataset loading, transforms, tokenization
├── inferencer.py                     # Interleaved text+image inference engine
│
├── configs/                          # Launch configurations
│   ├── train/                        #   SLURM training scripts
│   │   ├── train_thinkmorph.sh
│   │   ├── train_spatial_reasoning.sh
│   │   ├── train_text_reasoning.sh
│   │   ├── train_interleaved_reasoning.sh
│   │   ├── train_spatial_reasoning_multinode.sh
│   │   ├── train_spatial_reasoning_tamia_h200_sg.sh
│   │   ├── train_text_only_thinking_tamia_h200_sg.sh
│   │   ├── train_visual_only_thinking_tamia_h200_sg.sh
│   │   ├── train_qwen_spatial_reasoning.sh
│   │   ├── train_qwen_spatial_reasoning_no_thinking.sh
│   │   ├── train_qwen_spatial_counting_SG.sh
│   │   └── merge_qwen.sh
│   └── data/                         #   Dataset YAML configs
│       ├── spatial_reasoning.yaml
│       ├── interleaved_reasoning.yaml
│       ├── text_reasoning.yaml
│       ├── text_only_thinking_scence_graph.yaml
│       └── visual_only_thinking_scence_graph.yaml
│
├── SpatialUnderstanding/             # Spatial understanding pipeline
│   ├── data_creation/                #   Stages 1-3: data pipeline
│   │   ├── rendering/                #     Stage 1: Infinigen multi-view rendering
│   │   ├── annotation/               #     Stage 2 & 2.5: Gemini annotation + QA parsing
│   │   │   └── templates/            #       Jinja2 prompt templates
│   │   └── training_data/            #     Stage 3: Parquet/JSONL training data creation
│   │
│   └── eval/                         #   Evaluation & inference
│       ├── mmsi/                     #     MMSI-Bench evaluation
│       ├── spatial/                  #     Spatial collaboration (BAGEL + Gemini)
│       ├── generation/               #     Description-based view generation
│       ├── visworld/                 #     VisWorld benchmark
│       └── qwen/                     #     Qwen LoRA evaluation
│
├── tools/                            # Utilities
│   ├── wandb/                        #   W&B upload scripts
│   └── dataset_utils/                #   Dataset splitting, filtering, preprocessing
│
└── infinigen/                        # Legacy/deprecated rendering scripts
```

---

## Pipeline Overview

```
Stage 1: RENDER VIEWS (Infinigen + Blender)
  ├── Orbit views            (3 views around room center)
  ├── Rear-offset views      (wall-side V1, center-room V2, bridge)
  ├── Center view            (focused on reference object)
  └── Top-down ego-mark      (V1 + V2 + filtered orthographic top-down)
          │
          ▼
Stage 2: ANNOTATE (Gemini API)
  ├── Orbit anchor annotations
  ├── Center view annotations (v4)
  └── Parse annotations into Q&A format
          │
          ▼
Stage 3: CREATE TRAINING DATA
  ├── Rear view        → parquet
  ├── Orbit anchor     → parquet
  ├── Parsed QA        → parquet / JSONL (interleaved / text-only / visual-only)
  └── Top-down map     → parquet / JSONL
          │
          ▼
Stage 4: TRAIN
  └── FSDP distributed training with thinking modes
          │
          ▼
Stage 5: EVALUATE
  ├── Spatial collaboration (BAGEL / Gemini)
  ├── Description-based view generation
  ├── MMSI-Bench
  └── VisWorld benchmark
```

---

## Stage 1: Render Views

All rendering scripts are in `SpatialUnderstanding/data_creation/rendering/`. They use Blender via Infinigen scenes and submit SLURM array jobs.

### Orbit Views

Renders 3 orbit views around the room center with ~120 degree total angular span.

| | |
|---|---|
| **Script** | `data_creation/rendering/render_orbit_views.py` |
| **Batch** | `data_creation/rendering/batch_render_orbit_view.sh` |
| **Output** | `orbit_view_{0,1,2}_*.png` + `orbit_metadata_*.json` |

### Rear-Offset Views

Renders 3 views for spatial memory reasoning:
- **V1**: wall-side camera with wide view of a target object
- **V2**: center-room camera where the target object falls into the rear blind spot
- **Bridge**: same position as V2 but looking directly at the target object

| | |
|---|---|
| **Script** | `data_creation/rendering/render_rear_offset_views.py` |
| **Batch** | `data_creation/rendering/batch_render_rear_view.sh` |
| **Output** | `v1_*.png`, `v2_*.png`, `bridge_*.png` + `rear_offset_metadata_*.json` |

### Center View

Renders a single view focused on a reference object mentioned in a spatial question.

| | |
|---|---|
| **Script** | `data_creation/rendering/render_center_view_v4.py` |
| **Batch** | `data_creation/rendering/batch_render_center_views_v4.sh` |
| **Output** | `center_view_*.png` + `center_view_*_metadata.json` |

### Top-Down Ego-Mark Views

Renders 3 views for top-down layout reasoning:
- **V1**: wall-side camera establishing the global room boundary
- **V2**: center-room camera panned left/right, sharing at least one visible object with V1
- **Target**: orthographic top-down view with ego-mark (red sphere for V2 position, cyan cone for facing direction)

| | |
|---|---|
| **Script** | `data_creation/rendering/render_topdown_egomark_dataset.py` |
| **Batch** | `data_creation/rendering/generate_topdown_maps.sh` |
| **Output** | `v1_*.png`, `v2_*.png`, `target_topdown_*.png` + `topdown_egomark_metadata_*.json` |

---

## Stage 2: Annotate with Gemini

Scripts are in `SpatialUnderstanding/data_creation/annotation/`. All use Gemini API with extended thinking, resumable checkpointing, and 32 parallel workers.

### Orbit Anchor Annotations

Sends 3 orbit views to Gemini and generates a spatial description of the middle view.

| | |
|---|---|
| **Script** | `data_creation/annotation/run_gemini_annotator_orbit_anchor.py` |
| **Template** | `data_creation/annotation/templates/Gemini_annotator_prompt.jinja` |

### Center View Annotations (v4)

Sends 3 images (wall-side, center-room, center view) to Gemini with rear-view prompt template.

| | |
|---|---|
| **Script** | `data_creation/annotation/run_gemini_annotator_center_view_v4.py` |
| **Template** | `data_creation/annotation/templates/Gemini_annotator_prompt_rear_view.jinja` |

```bash
export VisualCoT_GEMINI=<your_api_key>
python data_creation/annotation/run_gemini_annotator_center_view_v4.py \
    --rendered_dir /path/to/rendered_v4_spatial \
    --output_file /path/to/annotations.json
```

### Parse Annotations into Q&A Format

Converts narrative annotations into structured question + thinking pairs.

| | |
|---|---|
| **Center view parser** | `data_creation/annotation/run_gemini_parser.py` |
| **Rear view parser** | `data_creation/annotation/run_gemini_parser_rear_view.py` |

```bash
python data_creation/annotation/run_gemini_parser.py \
    --input_file /path/to/annotations.json \
    --output_file /path/to/parsed_qa.json \
    --max_workers 32
```

---

## Stage 3: Create Training Data

Scripts are in `SpatialUnderstanding/data_creation/training_data/`. They convert annotations into Parquet or JSONL format.

### Rear View / Orbit Anchor Training Data

| | |
|---|---|
| **Rear view** | `data_creation/training_data/create_rear_view_training_data.py` |
| **Orbit anchor** | `data_creation/training_data/create_orbit_anchor_training_data.py` |

### Parsed QA Training Data (3 thinking modes)

| | |
|---|---|
| **Script** | `data_creation/training_data/create_training_data_from_parsed_qa.py` |

| Mode | Input | Output format |
|------|-------|---------------|
| `interleaved_thinking` | 2 images + center view + question | `<think>...</think><image_start>...<image_end>` (Parquet) |
| `text_only_thinking` | 2 images + question | `<think>...</think>` (JSONL ShareGPT) |
| `visual_only_thinking` | 2 images + center view + question | `<image_start>...<image_end>` (Parquet) |

```bash
python data_creation/training_data/create_training_data_from_parsed_qa.py \
    --annotation_file /path/to/parsed_qa.json \
    --output_dir /path/to/output \
    --thinking_mode interleaved_thinking \
    --max_workers 16
```

### Top-Down Map Training Data

| | |
|---|---|
| **Script** | `data_creation/training_data/create_training_data_from_topdown.py` |

Supports `visual_only` (Parquet) and `no_thinking` (JSONL) modes.

---

## Stage 4: Train

Training scripts are in `configs/train/`. All use `train/pretrain_unified_navit.py` with FSDP.

```bash
# Example: spatial reasoning training (4x A100)
sbatch configs/train/train_spatial_reasoning.sh

# Example: interleaved thinking (text + image reasoning)
sbatch configs/train/train_interleaved_reasoning.sh
```

Key env vars to configure in each script:
- `MODEL_PATH`: base BAGEL checkpoint
- `--dataset_config_file`: points to `configs/data/*.yaml`
- `--output_dir`: checkpoint save directory

---

## Stage 5: Evaluate

### Spatial Collaboration

BAGEL inference on spatial collaboration dataset with SLURM sharding.

| | |
|---|---|
| **Inference** | `eval/spatial/run_inference_bagel_spatial.py` |
| **SLURM (4-GPU)** | `eval/spatial/run_bagel_spatial_4gpu.sh` |
| **Gemini baseline** | `eval/spatial/run_inference_gemini.py` |
| **Gemini + topdown** | `eval/spatial/run_inference_gemini_topdown.py` |
| **Merge results** | `eval/spatial/merge_bagel_spatial_results.py` |
| **Text metrics** | `eval/spatial/eval_thinking_text_metrics.py` |

```bash
# Run 4-GPU inference
MODEL_PATH=/path/to/checkpoint \
DATASET_FILES="relative_dataset_V_Final_2000_test.json" \
sbatch eval/spatial/run_bagel_spatial_4gpu.sh
```

GPU variants: `run_bagel_spatial.sh`, `_1gpu.sh`, `_2gpu.sh`, `_4gpu.sh`, `_l40s.sh`

### Description-Based View Generation

| | |
|---|---|
| **SLURM scripts** | `eval/generation/run_bagel_center_view*.sh` |
| **Evaluation** | `eval/generation/evaluate_generation.py` |

Metrics: PSNR, SSIM, LPIPS, Depth (Scale-Invariant RMSE), LLM (Gemini instruction following).

```bash
python eval/generation/evaluate_generation.py \
    --results_dir /path/to/output \
    --metrics psnr ssim lpips depth llm
```

### MMSI-Bench

| | |
|---|---|
| **Inference** | `eval/mmsi/eval_mmsi_bench.py` |
| **SLURM** | `eval/mmsi/eval_mmsi_bench.sh` |
| **Merge + eval** | `eval/mmsi/merge_and_eval_mmsi.py` |

### VisWorld

| | |
|---|---|
| **Inference** | `eval/visworld/eval_visworld.py` |
| **SLURM** | `eval/visworld/eval_visworld.sh` |

---

## Quick Reference

| Stage | Task | Script |
|-------|------|--------|
| 1 | Render orbit views | `data_creation/rendering/render_orbit_views.py` |
| 1 | Render rear-offset views | `data_creation/rendering/render_rear_offset_views.py` |
| 1 | Render center view | `data_creation/rendering/render_center_view_v4.py` |
| 1 | Render top-down ego-mark | `data_creation/rendering/render_topdown_egomark_dataset.py` |
| 2 | Annotate orbit anchor | `data_creation/annotation/run_gemini_annotator_orbit_anchor.py` |
| 2 | Annotate center view (v4) | `data_creation/annotation/run_gemini_annotator_center_view_v4.py` |
| 2.5 | Parse center view annotations | `data_creation/annotation/run_gemini_parser.py` |
| 2.5 | Parse rear view annotations | `data_creation/annotation/run_gemini_parser_rear_view.py` |
| 3 | Create rear view training data | `data_creation/training_data/create_rear_view_training_data.py` |
| 3 | Create orbit anchor training data | `data_creation/training_data/create_orbit_anchor_training_data.py` |
| 3 | Create parsed QA training data | `data_creation/training_data/create_training_data_from_parsed_qa.py` |
| 3 | Create top-down training data | `data_creation/training_data/create_training_data_from_topdown.py` |
| 4 | Train (spatial reasoning) | `configs/train/train_spatial_reasoning.sh` |
| 4 | Train (interleaved thinking) | `configs/train/train_interleaved_reasoning.sh` |
| 5 | Eval spatial collaboration | `eval/spatial/run_bagel_spatial_4gpu.sh` |
| 5 | Eval view generation | `eval/generation/run_bagel_center_view.sh` |
| 5 | Eval generation quality | `eval/generation/evaluate_generation.py` |
| 5 | Eval MMSI-Bench | `eval/mmsi/eval_mmsi_bench.sh` |
| 5 | Eval VisWorld | `eval/visworld/eval_visworld.sh` |

All paths above are relative to `SpatialUnderstanding/` unless they start with `configs/` (repo root).
