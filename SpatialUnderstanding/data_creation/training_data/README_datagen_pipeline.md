# Question Generation Pipeline (`datagen_pipeline.py`)

End-to-end pipeline that takes 3D indoor scenes rendered by Infinigen and produces spatial reasoning multiple-choice QA datasets for a **two-agent collaboration task**.

---

## Overview

```
Infinigen (Blender)                    datagen_pipeline.py
┌──────────────────┐    ┌──────────────────────────────────────────────────┐
│ Render 1000      │    │  14-stage pipeline                               │
│ indoor scenes    │───>│  Scene filtering → Object/camera extraction →    │
│ with 2 cameras   │    │  LLM detection → Descriptions → Questions →     │
│ per scene        │    │  Paraphrasing → Aggregation → Filtering          │
└──────────────────┘    └──────────────────────────────────────────────────┘
                                         │
                                         ▼
                        5 spatial QA datasets + map questions
                        (counting, anchor, relative distance,
                         spatial orientation, perspective taking)
```

**Two-agent setup:**
- **Agent 1** — wall-side camera (`Image_0_0_0048_0.png`)
- **Agent 2** — center-room camera (`Image_1_0_0048_0.png`)

One agent receives a question and must collaborate with the other agent (who sees a different view) to answer it correctly.

---

## Scene Generation (Infinigen)

Shell scripts (e.g., `infinigen/diningroom_v1001_final_part7.sh`) submit SLURM jobs that render indoor scenes via Blender:

```bash
python -m infinigen.datagen.manage_jobs \
    --output_folder outputs/DiningRoom_v1001_Final_Part7 \
    --num_scenes 1000 \
    --configs singleroom.gin studio.gin \
    --pipeline_overrides \
        iterate_scene_tasks.n_camera_rigs=2 \
    --overrides \
        restrict_solving.restrict_parent_rooms=["DiningRoom"]
```

Key parameters:
- `n_camera_rigs=2` — renders 2 viewpoints per scene
- `restrict_parent_rooms` — controls room type (DiningRoom, Kitchen, Bedroom, Bathroom, LivingRoom)
- Each completed scene contains `frames/Image/camera_0/Image_{0,1}_0_0048_0.png`

---

## Pipeline Stages

Run via `--stages_to_run`. Stages are independent and can be run selectively.

### Stage 1: Scene Discovery (`find_all_scenes`)

Scans the base directory for complete scenes. A scene is "complete" when both camera images exist. Handles 4 directory layouts:
- Single scene
- Multiple scenes, one room type
- Multiple scenes, multiple room types
- Multiple folders, multiple scenes, multiple room types

### Stage 2: Scene Filtering (`scene_filtering`)

Uses an LLM (GPT-4o-mini via OpenAI API) to evaluate scene quality from rendered images. Each scene receives an `ACCEPT.txt` or `REJECT.txt` file with the reasoning.

| | |
|---|---|
| **Module** | `scene_filtering.filter_scenes()` |
| **Model** | GPT-4o-mini |
| **Output** | `ACCEPT.txt` or `REJECT.txt` per scene |

### Stage 3: Object Info Extraction (`scene_object_info`)

Runs a Blender Python script to extract object geometry and properties from `scene.blend`.

| | |
|---|---|
| **Script** | `question_generation_v2/get_object_info.py` (via Blender) |
| **Input** | `coarse/scene.blend` |
| **Output** | `visible_objects.json` |

### Stage 4: Camera Info Extraction (`scene_camera_info`)

Extracts camera poses (position, rotation, FOV) from the Blender scene.

| | |
|---|---|
| **Script** | `question_generation_v2/get_camera_info.py` (via Blender) |
| **Input** | `coarse/scene.blend` |
| **Output** | `cameras.json` |

### Stage 5: Blender Color Info (`scene_blender_color_info`)

Extracts ground-truth material/color information from Blender objects. Auto-unzips `scene.blend.zip` if needed.

| | |
|---|---|
| **Script** | `question_generation_v2/get_blender_color_v2.py` (via Blender) |
| **Input** | `coarse/scene.blend` |
| **Output** | `blender_colors.json` |

### Stage 6: LLM Visible Object Detection (`scene_llm_visible_objects`)

Uses a vision-language model to detect objects visible in the rendered images (both camera views).

| | |
|---|---|
| **Module** | `question_generation_v2.llm_visible_objects.run_visibility_check()` |
| **Model** | Qwen2.5-VL-3B-Instruct (via vLLM) |
| **Input** | Rendered images |
| **Output** | `llm_detected_objects.json` |

### Stage 7: Object Bounding (`scene_bound_objects`)

Creates bounding box visualizations for each detected object in both camera views.

| | |
|---|---|
| **Script** | `question_generation_v2/bound_objects.py` |
| **Input** | `llm_detected_objects.json` + rendered images |
| **Output** | `bounds/camera_0_0/` and `bounds/camera_1_0/` (includes `*_all_boxes.png`) |

### Stage 8: Color Detection (`scene_obj_color_info`)

Uses an LLM to identify object colors from the rendered images.

| | |
|---|---|
| **Module** | `question_generation_v2.get_color_info.run_color_detection()` |
| **Model** | GPT-4o-mini |
| **Input** | Rendered images + detected objects |
| **Output** | `llm_detected_objects_colors.json` |

### Stage 9: Description Generation (`scene_generate_descriptions`)

Generates natural language descriptions for each object by combining LLM-detected data with Blender ground truth.

| | |
|---|---|
| **Script** | `question_generation_v2/generate_descriptions.py` |
| **Input** | `llm_detected_objects_colors.json` + `visible_objects.json` |
| **Output** | `visible_objects_with_descriptions.json`, `full_description.json` |

### Stage 10: Perception Solving (`scene_solve_perception`)

Creates structured perception text for each agent, describing what they can see from their viewpoint.

| | |
|---|---|
| **Script** | `question_generation_v2/perception_solving_descriptions.py` |
| **Input** | `llm_detected_objects.json` |
| **Output** | `agent_1_input.txt`, `agent_2_input.txt` |

### Stage 11: Question Generation (`scene_generate_questions`)

Generates 5 types of spatial reasoning questions as multiple-choice (see [Question Types](#question-types) below).

| | |
|---|---|
| **Script** | `question_generation_v2/generate_questions.py` |
| **Input** | `visible_objects_with_descriptions.json`, `visible_objects.json`, `cameras.json` |
| **Output** | `questions.json` |

### Stage 12: Map Question Generation (`scene_generate_maps`)

Generates top-down map-based multiple-choice questions with visual map images.

| | |
|---|---|
| **Script** | `question_generation_v2/map_gen_v6.py` |
| **Input** | `visible_objects.json`, `cameras.json` |
| **Output** | `map_questions_v8.json`, `map_questions_ankur_v8/` (contains `map_questions_format1.json`, `map_questions_format2.json`, map images) |

Two answer formats:
- **Format 1**: Simple letter options (A, B, C, D)
- **Format 2**: Structured options with object categories and coordinates

### Stage 13: Question Paraphrasing (`scene_generate_paraphrase`)

LLM rephrases all questions for linguistic diversity.

| | |
|---|---|
| **Module** | `question_generation_v2.paraphrase_questions.paraphrase_across_scenes()` |
| **Model** | GPT-4o-mini |
| **Input** | `questions.json` |
| **Output** | `questions_paraphrased.json` |

### Stage 14a: Aggregation (`aggregate_data`)

Collects all paraphrased questions across scenes into 5 per-type dataset files:
- `dataset_counting_questions_{version}.json`
- `dataset_anchor_questions_{version}.json`
- `dataset_relative_distance_questions_{version}.json`
- `dataset_spatial_questions_{version}.json`
- `dataset_perspective_taking_questions_{version}.json`

### Stage 14b: Question Filtering (`filter_questions`)

Applies quality filters to remove trivial or unreliable questions (see [Filtering Rules](#filtering-rules)).

---

## Question Types

All questions are **multiple-choice with 4 options**, assigned to one of the two agents.

### 1. Counting
> "How many [object] are in the scene?"

- Difficulty metrics: `difficulty_sum` (total across views), `difficulty_int`
- Answer is an integer count

### 2. Anchor
> Identify a reference object based on a spatial description.

- Difficulty metrics: `difficulty` (overall), `description_difficulty`, `distractor_difficulty`
- Includes `option_categories` for distractor types

### 3. Relative Distance
> Compare distances between objects from a specific viewpoint.

- Difficulty metrics: `difficulty`, `description_difficulty`
- Includes `option_distances`, `ans_present_in_view`, `agent_distribution`

### 4. Spatial Orientation
> Reason about the direction or angle to an object.

- Metadata: `angle`, `distance`, `other_agent_angle`, `other_agent_distance`
- Difficulty derived from spatial separation

### 5. Perspective Taking
> Answer a spatial question from the partner agent's viewpoint.

- Same metadata as spatial orientation
- Tests ability to reason about what the other agent sees

### 6. Map Questions (separate from main 5)
> Given a top-down map, identify the correct location/configuration.

- Generated per-agent with distractor types: counting, type2 (swapped), type3 (swapped)
- Includes generated map images

---

## Filtering Rules

### Counting Questions
Skip if the question is trivially answerable from the room type:
- Sink/Bathtub/Toilet/Mirror in Bathroom
- Shelf/Cabinet/Oven/Beverage Fridge/Dishwasher/Sink in Kitchen
- Table Dining/Chair in DiningRoom

Also skip when `difficulty_sum == answer_value` (no information gain from collaboration).

### Spatial Orientation / Perspective Taking / Relative Distance
Skip when `distance < 1.5` meters (objects too close for meaningful spatial reasoning).

---

## Output Sample Format

Each question sample in the aggregated dataset contains:

```json
{
  "sample_id": "counting_000042",
  "question_type": "counting",
  "room_part": "DiningRoom_v1001_Final_Part7",
  "scene_id": "scene_00123",
  "user_1_image_local_path": "/path/to/frames/Image/camera_0/Image_0_0_0048_0.png",
  "user_2_image_local_path": "/path/to/frames/Image/camera_0/Image_1_0_0048_0.png",
  "user_1_goal": "Communicate with your partner to answer the following question correctly.",
  "user_2_goal": "Communicate with your partner to help them answer their question correctly.",
  "user_1_question": "How many chairs are visible in total?",
  "user_2_question": null,
  "options_user_1": ["2", "3", "4", "5"],
  "options_user_2": null,
  "user_1_gt_answer_idx": 1,
  "user_1_gt_answer_text": "3",
  "correct_answer": "3",
  "question_both_views": "How many chairs are visible across both views?",
  "scene_intersection": ["chair_1", "table_1"],
  "scene_union": ["chair_1", "chair_2", "chair_3", "table_1", "lamp_1"],
  "user_1_perception": "/path/to/agent_1_input.txt",
  "user_2_perception": "/path/to/agent_2_input.txt"
}
```

---

## Models Used

| Task | Model | API |
|------|-------|-----|
| Scene filtering | GPT-4o-mini | OpenAI |
| Visible object detection | Qwen2.5-VL-3B-Instruct | vLLM (local) |
| Color detection | GPT-4o-mini | OpenAI |
| Paraphrasing | GPT-4o-mini | OpenAI |

---

## Usage

```bash
# Full pipeline (all stages)
python datagen_pipeline.py \
    --base_dir /path/to/infinigen/outputs \
    --scene_datafile ./dataset.json \
    --stages_to_run scene_filtering scene_object_info scene_camera_info \
        scene_blender_color_info scene_llm_visible_objects scene_bound_objects \
        scene_obj_color_info scene_generate_descriptions scene_solve_perception \
        scene_generate_questions scene_generate_maps scene_generate_paraphrase \
        aggregate_data filter_questions \
    --max_scenes 100 \
    --log_wandb

# Run only question generation (assumes earlier stages are complete)
python datagen_pipeline.py \
    --base_dir /path/to/infinigen/outputs \
    --stages_to_run scene_generate_questions scene_generate_paraphrase aggregate_data

# Run only map generation
python datagen_pipeline.py \
    --base_dir /path/to/infinigen/outputs \
    --stages_to_run scene_generate_maps
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--base_dir` | Root directory containing rendered scenes | Required |
| `--scene_datafile` | Output JSON tracking all scene metadata | `./dataset_checkupv1_ankur.json` |
| `--stages_to_run` | Which pipeline stages to execute | `["scene_generate_maps"]` |
| `--max_scenes` | Limit number of scenes (random sample) | `None` (all) |
| `--seed` | Random seed for scene sampling | `42` |
| `--dry_run` | Print commands without executing | `False` |
| `--log_wandb` | Log results to W&B | `False` |
| `--overwrite_files` | Re-run stages even if outputs exist | `False` |
| `--question_version` | Version tag for output filenames | `V4` |

---

## Per-Scene File Structure

After a full pipeline run, each scene directory contains:

```
scene_XXXXX/
├── frames/Image/camera_0/
│   ├── Image_0_0_0048_0.png          # Agent 1 view
│   └── Image_1_0_0048_0.png          # Agent 2 view
├── coarse/
│   ├── scene.blend                    # Blender scene file
│   └── asset_parameters.json
├── visible_objects.json               # Blender ground-truth objects
├── cameras.json                       # Camera poses
├── blender_colors.json                # Ground-truth material colors
├── llm_detected_objects.json          # Vision model detections
├── llm_detected_objects_colors.json   # LLM-identified colors
├── visible_objects_with_descriptions.json  # NL descriptions
├── full_description.json              # Full scene description
├── agent_1_input.txt                  # Agent 1 perception text
├── agent_2_input.txt                  # Agent 2 perception text
├── questions.json                     # Generated questions (5 types)
├── questions_paraphrased.json         # Paraphrased questions
├── map_questions_v8.json              # Map questions
├── map_questions_ankur_v8/            # Map question details
│   ├── map_questions_format1.json
│   ├── map_questions_format2.json
│   └── *.png                          # Map images
├── bounds/
│   ├── camera_0_0/camera_0_0_all_boxes.png
│   └── camera_1_0/camera_1_0_all_boxes.png
├── ACCEPT.txt or REJECT.txt           # Scene filtering result
└── logs/
```

---

## Data Flow Diagram

```
                    Infinigen (Blender rendering)
                              │
                              ▼
                    2 rendered images per scene
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        scene.blend     Image_0.png     Image_1.png
              │               │               │
    ┌─────────┴─────────┐    └───────┬───────┘
    ▼                   ▼            ▼
visible_objects.json  cameras.json  llm_detected_objects.json
blender_colors.json                 llm_detected_objects_colors.json
    │                   │            │
    └─────────┬─────────┘            │
              ▼                      │
    visible_objects_with_descriptions.json
              │                      │
              ├──────────────────────┘
              ▼
    questions.json (5 types) ──► questions_paraphrased.json
    map_questions_v8.json              │
              │                        │
              └────────┬───────────────┘
                       ▼
              Aggregated datasets (per question type)
                       │
                       ▼
              Filtered datasets (quality filters applied)
```
