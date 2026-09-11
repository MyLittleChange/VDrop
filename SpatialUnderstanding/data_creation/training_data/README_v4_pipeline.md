# V4 Training Data Pipeline

Pipeline to go from V4 QA JSONs → normalized datasets → panorama renders → mixed SFT training data.

## Overview

| Step | Script | Output |
|------|--------|--------|
| 1. Normalize | `tools/dataset_utils/normalize_v4_datasets.py` | `dataset_*_V4_normalized.json` |
| 2. Filter test set | `tools/dataset_utils/filter_test_set_v4.py` | `approved_*_filtered_v4.json` |
| 3. Prepare render split | `prepare_panorama_v4_split.py` | `panorama_v4_scenes.json` |
| 4. Render panoramas | `rendering/batch_render_panorama_v4.sh` | `rendered_panorama_v4/<scene>/...png` |
| 5. Filter rendered | `prepare_panorama_v4_split.py --filter_rendered` | `panorama_v4_scenes_filtered.json` |
| 6. Create SFT data | `create_mix_v4_sft_data.py` | `training_data_mix/{mode}/` |

---

## Prerequisites

- V4 filtered QA JSONs in `/path/to/scratch/infinigen/`:
  - `dataset_counting_questions_filtered_V4.json` (306 samples)
  - `dataset_anchor_questions_filtered_V4.json` (93 samples)
  - `dataset_spatial_questions_filtered_V4.json` (799 samples)
  - `dataset_relative_distance_questions_filtered_V4.json` (25 samples)
  - `dataset_perspective_taking_questions_filtered_V4.json` (802 samples)
- Existing training panoramas already rendered in `rendered_panorama_train/` (1876 samples, 653 scenes)

---

## Step 1: Normalize V4 datasets

```bash
cd /path/to/ThinkMorph-BAGEL-release
python tools/dataset_utils/normalize_v4_datasets.py
```

- Swaps user_1/user_2 perspectives so the question is always from user_2's view
- Adds `topdown_path: null` and `panorama_path: null` fields
- Output: `dataset_*_filtered_V4_normalized.json` (same directory)

---

## Step 2: Filter test set (confirm no overlap)

```bash
python tools/dataset_utils/filter_test_set_v4.py
```

- Checks approved test sets for any scenes in V4 training data (expected: 0 overlap)
- Output: `approved_*_normalized_filtered_v4.json` in `spatial_collab_dataset/`

---

## Step 3: Prepare panorama rendering split

```bash
python SpatialUnderstanding/data_creation/training_data/prepare_panorama_v4_split.py
```

- Collects unique scenes from all 5 normalized V4 datasets (~315 unique scenes)
- Excludes test scenes
- Output:
  - `/path/to/scratch/infinigen/panorama_v4_scenes.json`
  - `/path/to/scratch/infinigen/scene_to_sample_id_v4.json`

---

## Step 4: Render panoramas

```bash
# Check total unique scenes from step 3 output, then:
bash SpatialUnderstanding/data_creation/rendering/batch_render_panorama_v4.sh \
    /path/to/scratch/infinigen/panorama_v4_scenes.json \
    315   # update with actual scene count
```

- SLURM array job, `long` partition, 8 CPUs, 32Gb, 5h per task
- Output: `/path/to/scratch/infinigen/rendered_panorama_v4/<scene_id>/panorama_blender_limits_*.png`

**Note:** V4 scenes use `fine/scene.blend` (not `coarse/`). The `BLEND_BASE_PATH` env var
is set automatically by the script. If the render script can't find blend files,
check that `BLEND_BASE_PATH=/path/to/scratch/infinigen/outputs_rendered`.

---

## Step 5: Filter to successfully rendered scenes

```bash
python SpatialUnderstanding/data_creation/training_data/prepare_panorama_v4_split.py \
    --filter_rendered \
    --panorama_dir /path/to/scratch/infinigen/rendered_panorama_v4
```

- Output:
  - `panorama_v4_scenes_filtered.json`
  - `scene_to_sample_id_v4_filtered.json`

---

## Step 6: Create mixed SFT training data

### Visual-only mode (panorama reasoning image, Parquet output)

```bash
python SpatialUnderstanding/data_creation/training_data/create_mix_v4_sft_data.py \
    --thinking_mode visual_only
```

- Mixes 1876 existing + up to 2025 V4 samples (skips V4 without rendered panorama)
- Output: `/path/to/scratch/infinigen/training_data_mix/visual_only/chunk_*.parquet`

### No-thinking mode (direct answer, JSONL output)

```bash
python SpatialUnderstanding/data_creation/training_data/create_mix_v4_sft_data.py \
    --thinking_mode no_thinking
```

- All samples included (no panorama needed)
- Output: `/path/to/scratch/infinigen/training_data_mix/no_thinking/no_thinking.jsonl`

---

## Expected final counts

| Dataset | Existing | V4 | Total |
|---------|----------|----|-------|
| Samples (no_thinking) | 1876 | 2025 | ~3901 |
| Samples (visual_only) | 1876 | ~315 (one per scene) | ~2191 |

---

## Troubleshooting

**Blend file not found during panorama render:**
- V4 scenes use `fine/scene.blend` not `coarse/scene.blend.zip`
- Verify `BLEND_BASE_PATH` is set to `/path/to/scratch/infinigen/outputs_rendered`
- The render script looks up `BLEND_BASE_PATH/<room_part>/<scene_id>/fine/scene.blend`

**Panorama path not resolved:**
- Check `scene_to_sample_id_v4_filtered.json` exists (run step 5 first)
- Verify `rendered_panorama_v4/<scene_id>/panorama_blender_limits_<sample_id>.png` exists

**Missing V4 normalized files:**
- Run step 1 (normalize) before steps 3-6
