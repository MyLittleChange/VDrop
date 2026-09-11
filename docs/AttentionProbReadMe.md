# Answer-Token Attention Probe

This diagnostic checks whether BAGEL attends to its generated visual-thinking
bridge image when it predicts the final answer token.

The main implementation is in:

- `tools/answer_attention_probe.py`: single-sample probe.
- `tools/answer_attention_probe_batch.py`: multi-sample runner that keeps one model loaded.
- `tools/run_bridge_mask_attention_100.sh`: fixed 100-sample comparison between visual-only and bridge-mask checkpoints.
- `debug_attention.py`: FlashAttention Q/K capture, attention reconstruction, span summaries, and heatmap extraction.
- `inferencer.py` and `modeling/bagel/bagel.py`: callback path for text-generation attention capture.

## Why This Exists

The concern is that visual-thinking training may not actually make the model use
the generated bridge image. If visual-thinking accuracy is close to no-thinking
accuracy, we need a direct diagnostic:

1. Generate the visual-thinking bridge image.
2. Feed that bridge back into BAGEL.
3. Capture answer-token attention during final answer decoding.
4. Measure how much attention mass goes to the bridge image tokens versus
   original input views and text.
5. Render heatmaps over the generated bridge image.

The probe does not prove causal necessity by itself, but it tells us whether the
answer-token query is looking at the bridge tokens at all.

## High-Level Inference Flow

For the visual-thinking path, one probe sample runs as:

1. Load `V1`, `V2`, question, options, and ground-truth answer from the spatial
   dataset.
2. Initialize BAGEL generation context.
3. Add optional generic visual-thinking system text.
4. Add `V1` as VAE tokens and ViT tokens.
5. Add `V2` as VAE tokens and ViT tokens.
6. Add the question/options prompt.
7. Generate the first text segment. This is usually visual planning text and may
   contain `<image_start>`.
8. If bridge generation is enabled, append that generated visual-plan text back
   into the context.
9. Generate a bridge image from the updated context.
10. Choose which bridge image to feed back according to `--bridge_mode`.
11. Add the selected bridge image as VAE tokens and ViT tokens.
12. Generate final answer text while capturing attention Q/K tensors.
13. Find the generation step that predicted the answer token, for example the
    token `B` in `<answer>B</answer>`.
14. Summarize attention mass to named spans and render image-token heatmaps.

The important detail is that `inferencer.gen_text(...)` returns generated text
and a token trace, but it does not mutate `gen_context`. Therefore, before image
generation, `first_text` must be added back with:

```python
gen_context = add_text_with_span(
    inferencer,
    gen_context,
    first_text,
    spans,
    "visual_plan_text",
)
```

Without this, bridge generation would not be conditioned on the model's own
visual plan, and the span ledger would not know where the visual-plan text lives
in the KV cache.

## Bridge Modes

`--bridge_mode` controls what image is inserted before final answer decoding:

- `normal`: generate the bridge image and feed that generated image back.
- `blank`: generate the bridge image for logging, but feed a gray blank image.
- `swap`: generate the bridge image for logging, but feed `--swap_bridge_path`.
- `none`: do not feed a bridge image.

For `normal`, `blank`, and `swap`, attention capture happens during the final
answer decode after the bridge image has been appended.

For `none`, there is no bridge insertion stage, so the first text decode may be
the answer decode. The probe therefore captures first-text attention when
`bridge_mode == "none"`.

## What V1 And V2 Mean

The spatial dataset provides two views:

- `V1`: `answerer_image`, the first/user-1 view.
- `V2`: `helper_image`, the second/user-2 partner view.

Both are inserted into the BAGEL context as two token spans each:

- `V1_vae`, `V2_vae`: image-generation latent-space tokens.
- `V1_vit`, `V2_vit`: visual-understanding ViT tokens.

The bridge image is also inserted as:

- `bridge_vae`
- `bridge_vit`

## Attention Capture

BAGEL uses fused FlashAttention, so regular `output_attentions=True` is not
enough. The probe uses the monkey-patch path in `debug_attention.py` to capture
Q/K tensors from selected layers during text decoding.

For each captured answer step and layer:

1. Reconstruct attention scores from captured Q/K.
2. Apply the causal mask.
3. Softmax over key positions.
4. Average attention across heads.
5. Sum attention mass into named spans.

The key span groups are:

- `V1`
- `V2`
- `bridge_vae`
- `bridge_vit`
- `bridge_all`
- `text`
- `answer_prefix`
- `current_query`
- `other_unnamed`
- `total`

`other_unnamed` is the attention mass not covered by the named spans. It can
include special tokens, boundary tokens, BOS/EOS-like tokens, and any KV
positions not assigned to the explicit ledger.

## Answer-Token Step

In causal decoding, the query at step `t` predicts the next generated token.
The UI and metadata label this as the prediction step for the answer token.

The probe searches the decoded token trace for patterns like:

- `<answer>A</answer>`
- `Final Answer: A`
- fallback single-token `A`, `B`, `C`, or `D`

If it finds an answer token, attention is summarized at that step. If no answer
token is found, the probe falls back to the last generated token step.

## Output Files

A single probe output directory contains:

- `input_V1.png`, `input_V2.png`: input views.
- `generated_bridge.png`: bridge image generated by BAGEL.
- `bridge_used_for_answer.png`: actual bridge inserted before answer decoding.
- `final_text.txt`: final answer text.
- `all_text_outputs.txt`: first text plus final text.
- `metadata.json`: sample/checkpoint/answer/capture metadata.
- `spans.json`: named KV spans for text and image tokens.
- `answer_token_spans.json`: generated answer query positions.
- `token_trace.json`: generated token trace.
- `attention_spans.csv`: attention mass per individual span.
- `attention_groups.csv`: attention mass per group.
- `attention_summary.json`: JSON version of span/group summaries.
- `attention_groups_step*_layer*.png`: bar chart for span groups.
- `overlay_step*_layer*_bridge_vae.png`: bridge VAE heatmap overlay.
- `overlay_step*_layer*_bridge_vit.png`: bridge ViT heatmap overlay.
- optional `raw_qk/`: raw captured Q/K tensors unless `--skip_raw_qk` is used.

For large runs, `--skip_raw_qk` is recommended because raw Q/K tensors are large.
The CSV summaries and PNG heatmaps are still produced.

## Single-Sample Usage

Example:

```bash
/path/to/scratch/morph_env/bin/python tools/answer_attention_probe.py \
  --model_path /path/to/scratch/VisualCoT/BAGEL_checkpoints/BAGEL_format_training_data_mix_all_balance_visual_only_lora_7k \
  --dataset_json /path/to/scratch/spatial_collab_dataset/anchor_dataset_V_Final_2000.json \
  --sample_index 0 \
  --thinking_mode visual_only_thinking \
  --bridge_mode normal \
  --layers 27 \
  --num_timesteps 2 \
  --max_think_token_n 64 \
  --max_answer_token_n 32 \
  --max_mem_per_gpu 75GiB \
  --force_bridge \
  --no_think \
  --skip_raw_qk \
  --output_dir artifacts/answer_attention_probe/sample_0
```

Notes:

- `--force_bridge` forces bridge generation even if the first text does not emit
  `<image_start>`.
- `--no_think` disables the generic think-system prefix, but still allows the
  visual-thinking prompt/mode to drive bridge generation.
- `--layers 27` captures only one late layer. More layers give more signal but
  increase runtime and memory.

## Batch Usage

The batch runner loads one checkpoint once, then probes many samples from a
manifest:

```bash
/path/to/scratch/morph_env/bin/python tools/answer_attention_probe_batch.py \
  --model_path /path/to/checkpoint \
  --dataset_json /path/to/scratch/spatial_collab_dataset/anchor_dataset_V_Final_2000.json \
  --manifest artifacts/bridge_mask_attention_100/fixed_samples_seed20260504_n100.json \
  --run_name visual_only_lora_7k \
  --output_root artifacts/bridge_mask_attention_100 \
  --thinking_mode visual_only_thinking \
  --bridge_mode normal \
  --layers 27 \
  --max_mem_per_gpu 75GiB \
  --num_timesteps 2 \
  --max_think_token_n 64 \
  --max_answer_token_n 32 \
  --force_bridge \
  --no_think \
  --skip_raw_qk \
  --continue_on_error
```

This writes per-sample outputs under:

```text
artifacts/bridge_mask_attention_100/runs/<run_name>/sample_<sample_index>/
```

It also writes:

```text
artifacts/bridge_mask_attention_100/runs/<run_name>/batch_status.json
```

## Fixed 100-Sample Bridge-Mask Comparison

Use:

```bash
tools/run_bridge_mask_attention_100.sh
```

Default comparison:

- baseline: `BAGEL_format_training_data_mix_all_balance_visual_only_lora_7k`
- candidate: `BAGEL_format_mix_all_balance_visual_only_bridge_masked_lora_7k`

Default output:

```text
artifacts/bridge_mask_attention_100/
```

Important environment overrides:

```bash
SAMPLE_COUNT=100
SAMPLE_SEED=20260504
LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27"
NUM_TIMESTEPS=2
MAX_THINK_TOKEN_N=64
MAX_ANSWER_TOKEN_N=32
SAVE_RAW_QK=0
FORCE_RERUN=0
RESAMPLE=0
CONTINUE_ON_ERROR=1
```

For a smoke test:

```bash
OUTPUT_ROOT=/tmp/bridge_mask_attention_smoke \
SAMPLE_COUNT=1 \
SAMPLE_LIMIT=1 \
RESAMPLE=1 \
FORCE_RERUN=1 \
tools/run_bridge_mask_attention_100.sh
```

The comparison script creates or reuses a fixed manifest:

```text
fixed_samples_seed20260504_n100.json
```

The same sample indices are used for both checkpoints.

## Aggregated Comparison Outputs

The 100-sample script writes:

- `attention_distribution_wide.csv`: one row per run/sample/layer/answer step.
- `attention_distribution_summary.csv`: mean/std/median/p25/p75 per group.
- `attention_distribution_paired_deltas.csv`: per-sample candidate-minus-baseline
  deltas.
- `attention_distribution_delta_summary.csv`: summary of paired deltas.
- `attention_distribution_summary.md`: markdown table for quick comparison.
- `attention_distribution_failures.csv`: only present when samples are missing
  or failed.

The most important rows are usually:

- `bridge_all`: total attention mass to bridge VAE plus bridge ViT tokens.
- `bridge_vae`: attention to generated bridge latent/image-generation tokens.
- `bridge_vit`: attention to generated bridge visual-understanding tokens.
- `V1`, `V2`: attention to original input views.
- `text`: attention to prompt and generated visual-plan text.
- `other_unnamed`: mass outside explicitly named spans.

## Interpreting Results

If bridge attention is high:

- The answer token is at least looking at bridge tokens.
- Check heatmaps to see whether attention is spatially meaningful.
- Compare with `blank`, `swap`, and `none` bridge modes for necessity.

If bridge attention is low:

- The model may be answering mostly from text or original views.
- Visual-thinking may still help indirectly through text planning, so inspect
  `text` and `visual_plan_text` span attention.
- Run bridge ablations to test whether answer accuracy changes when the bridge
  is blanked or swapped.

If `other_unnamed` is large:

- Not all KV tokens are named in the span ledger.
- This is expected to some degree because special/boundary/current tokens are
  outside the main named spans.
- Use `covered_total` to see how much mass the explicit ledger accounts for.

## Practical Defaults

Recommended first-pass settings:

- `--layers 27`
- `--num_timesteps 2`
- `--max_think_token_n 64`
- `--max_answer_token_n 32`
- `--skip_raw_qk`
- `--force_bridge`

For deeper analysis, run late layers:

```bash
LAYERS="24 25 26 27" tools/run_bridge_mask_attention_100.sh
```

The current 100-sample comparison script records all BAGEL text-decoder layers
by default. Override `LAYERS` only when you want a faster or smaller diagnostic.
