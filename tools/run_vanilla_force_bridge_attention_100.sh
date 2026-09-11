#!/usr/bin/env bash
#SBATCH --job-name=vanilla_bridge_attn_100
#SBATCH --output=/path/to/ThinkMorph-BAGEL-release/artifacts/vanilla_force_bridge_attention_100_output.txt
#SBATCH --error=/path/to/ThinkMorph-BAGEL-release/artifacts/vanilla_force_bridge_attention_100_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=8:00:00

set -euo pipefail

# Fixed 100-sample answer-token attention run:
#   vanilla BAGEL-7B-MoT with forced bridge image generation
#
# By default this reuses the same fixed manifest path as
# run_bridge_mask_attention_100.sh, so vanilla/LoRA diagnostics are aligned by
# sample_index. Set MANIFEST=... explicitly if your comparison run used a custom
# manifest location.

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/path/to/ThinkMorph-BAGEL-release}}"
if [[ ! -f "$REPO_ROOT/tools/answer_attention_probe_batch.py" ]]; then
  REPO_ROOT="/path/to/ThinkMorph-BAGEL-release"
fi
cd "$REPO_ROOT"

SCRATCH_ROOT="${SCRATCH:-/path/to/scratch}"
PYTHON="${PYTHON:-/path/to/scratch/morph_env/bin/python}"
DATASET_JSON="${DATASET_JSON:-/path/to/scratch/spatial_collab_dataset/anchor_dataset_V_Final_2000.json}"

VANILLA_MODEL_PATH="${VANILLA_MODEL_PATH:-/path/to/scratch/models/BAGEL-7B-MoT}"
BRIDGE_COMPARE_OUTPUT_ROOT="${BRIDGE_COMPARE_OUTPUT_ROOT:-$SCRATCH_ROOT/VisualCoT/artifacts/bridge_mask_attention_100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH_ROOT/VisualCoT/artifacts/vanilla_force_bridge_attention_100}"

SAMPLE_COUNT="${SAMPLE_COUNT:-100}"
SAMPLE_SEED="${SAMPLE_SEED:-20260504}"
MANIFEST="${MANIFEST:-$BRIDGE_COMPARE_OUTPUT_ROOT/fixed_samples_seed${SAMPLE_SEED}_n${SAMPLE_COUNT}.json}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"

# BAGEL-7B-MoT text decoder has 28 layers, indexed 0..27.
LAYERS="${LAYERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27}"
MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-75GiB}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-2}"
MAX_THINK_TOKEN_N="${MAX_THINK_TOKEN_N:-64}"
MAX_ANSWER_TOKEN_N="${MAX_ANSWER_TOKEN_N:-32}"
VIT_MIN_SIZE="${VIT_MIN_SIZE:-512}"
TEXT_TEMPERATURE="${TEXT_TEMPERATURE:-0.3}"

RUN_NAME="${RUN_NAME:-vanilla_bagel_force_bridge}"
RUN_PROBES="${RUN_PROBES:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RESAMPLE="${RESAMPLE:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
SAVE_RAW_QK="${SAVE_RAW_QK:-0}"

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/runs" "$(dirname "$MANIFEST")"

if [[ "$RESAMPLE" == "1" ]]; then
  rm -f "$MANIFEST"
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "Creating fixed sample manifest: $MANIFEST"
  "$PYTHON" - "$DATASET_JSON" "$MANIFEST" "$SAMPLE_COUNT" "$SAMPLE_SEED" <<'PY'
import json
import random
import sys
from pathlib import Path

dataset_json = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
sample_count = int(sys.argv[3])
seed = int(sys.argv[4])

data = json.loads(dataset_json.read_text())
if sample_count > len(data):
    raise SystemExit(f"sample_count={sample_count} exceeds dataset length={len(data)}")

rng = random.Random(seed)
indices = rng.sample(range(len(data)), sample_count)
samples = []
for idx in indices:
    item = data[idx]
    samples.append({
        "sample_index": idx,
        "sample_id": item.get("sample_id", f"index_{idx}"),
        "question_type": item.get("question_type", ""),
    })

manifest = {
    "dataset_json": str(dataset_json),
    "dataset_len": len(data),
    "sample_count": sample_count,
    "seed": seed,
    "samples": samples,
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Wrote {manifest_path} with {len(samples)} fixed samples")
PY
else
  echo "Reusing fixed sample manifest: $MANIFEST"
fi

read -r -a LAYER_ARGS <<< "$LAYERS"

if [[ "$RUN_PROBES" == "1" ]]; then
  log_path="$OUTPUT_ROOT/logs/${RUN_NAME}_batch.log"
  cmd=(
    "$PYTHON" "$REPO_ROOT/tools/answer_attention_probe_batch.py"
    --model_path "$VANILLA_MODEL_PATH"
    --dataset_json "$DATASET_JSON"
    --manifest "$MANIFEST"
    --run_name "$RUN_NAME"
    --output_root "$OUTPUT_ROOT"
    --thinking_mode visual_only_thinking
    --bridge_mode normal
    --layers "${LAYER_ARGS[@]}"
    --max_mem_per_gpu "$MAX_MEM_PER_GPU"
    --vit_min_size "$VIT_MIN_SIZE"
    --num_timesteps "$NUM_TIMESTEPS"
    --max_think_token_n "$MAX_THINK_TOKEN_N"
    --max_answer_token_n "$MAX_ANSWER_TOKEN_N"
    --text_temperature "$TEXT_TEMPERATURE"
    --force_bridge
  )
  if [[ -n "$SAMPLE_LIMIT" ]]; then
    cmd+=(--sample_limit "$SAMPLE_LIMIT")
  fi
  if [[ "$FORCE_RERUN" == "1" ]]; then
    cmd+=(--force_rerun)
  fi
  if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
    cmd+=(--continue_on_error)
  fi
  if [[ "$SAVE_RAW_QK" != "1" ]]; then
    cmd+=(--skip_raw_qk)
  fi

  echo "RUN batch $RUN_NAME; log: $log_path"
  if ! "${cmd[@]}" >"$log_path" 2>&1; then
    echo "FAILED batch $RUN_NAME; log: $log_path" >&2
    tail -n 80 "$log_path" >&2 || true
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit 1
    fi
  fi
fi

echo "Aggregating vanilla forced-bridge attention distributions..."
"$PYTHON" - "$OUTPUT_ROOT" "$MANIFEST" "$SAMPLE_LIMIT" "$RUN_NAME" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
sample_limit_raw = sys.argv[3]
run_name = sys.argv[4]
manifest = json.loads(manifest_path.read_text())
manifest_samples = list(manifest["samples"])
if sample_limit_raw:
    manifest_samples = manifest_samples[:int(sample_limit_raw)]
sample_lookup = {int(item["sample_index"]): item for item in manifest_samples}

groups = [
    "V1",
    "V2",
    "bridge_vae",
    "bridge_vit",
    "bridge_all",
    "text",
    "answer_prefix",
    "current_query",
    "named_total",
    "covered_total",
    "other_unnamed",
    "total",
]

def pctile(values, q):
    values = sorted(values)
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac

wide_rows = []
failures = []
run_root = output_root / "runs" / run_name
for sample_index, sample in sample_lookup.items():
    sample_dir = run_root / f"sample_{sample_index}"
    meta_path = sample_dir / "metadata.json"
    groups_path = sample_dir / "attention_groups.csv"
    if not meta_path.exists() or not groups_path.exists():
        failures.append({
            "run": run_name,
            "sample_index": sample_index,
            "sample_id": sample.get("sample_id", ""),
            "reason": "missing metadata or attention_groups",
        })
        continue
    meta = json.loads(meta_path.read_text())
    keyed = {}
    rows = list(csv.DictReader(groups_path.open()))
    for row in rows:
        key = (int(row["step"]), int(row["layer"]))
        keyed.setdefault(key, {})
        keyed[key][row["group"]] = float(row["attention_mass"])
    for (step, layer), values in keyed.items():
        out = {
            "run": run_name,
            "checkpoint": meta.get("model_name", ""),
            "sample_index": sample_index,
            "sample_id": meta.get("sample_id", sample.get("sample_id", "")),
            "question_type": meta.get("question_type", sample.get("question_type", "")),
            "predicted_answer": meta.get("predicted_answer"),
            "correct_answer_idx": meta.get("correct_answer_idx"),
            "correct_answer": meta.get("correct_answer"),
            "target_step_source": meta.get("target_step_source", ""),
            "step": step,
            "layer": layer,
        }
        for group in groups:
            out[group] = values.get(group, "")
        wide_rows.append(out)

wide_path = output_root / "attention_distribution_wide.csv"
if wide_rows:
    with wide_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(wide_rows[0].keys()))
        writer.writeheader()
        writer.writerows(wide_rows)

summary_rows = []
for layer in sorted({row["layer"] for row in wide_rows}):
    layer_rows = [row for row in wide_rows if row["layer"] == layer]
    for group in groups:
        vals = [row[group] for row in layer_rows if row[group] != ""]
        if not vals:
            continue
        summary_rows.append({
            "run": run_name,
            "layer": layer,
            "group": group,
            "n": len(vals),
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "median": statistics.median(vals),
            "p25": pctile(vals, 0.25),
            "p75": pctile(vals, 0.75),
        })

summary_path = output_root / "attention_distribution_summary.csv"
if summary_rows:
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

def fmt_pct(value):
    return f"{100 * value:.2f}%"

md_lines = [
    "# Vanilla BAGEL Forced-Bridge Attention Distribution",
    "",
    f"Dataset: `{manifest['dataset_json']}`",
    f"Fixed sample seed: `{manifest['seed']}`",
    f"Requested samples: `{len(manifest_samples)}`",
    f"Run: `{run_name}`",
    "",
]
if summary_rows:
    md_lines.append("| layer | group | n | mean | median | p25 | p75 |")
    md_lines.append("|---:|:---|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        md_lines.append(
            f"| {row['layer']} | `{row['group']}` | {row['n']} | "
            f"{fmt_pct(row['mean'])} | {fmt_pct(row['median'])} | "
            f"{fmt_pct(row['p25'])} | {fmt_pct(row['p75'])} |"
        )

if failures:
    fail_path = output_root / "attention_distribution_failures.csv"
    with fail_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(failures[0].keys()))
        writer.writeheader()
        writer.writerows(failures)
    md_lines.extend(["", f"Missing/failed rows: `{len(failures)}`. See `{fail_path}`."])

md_path = output_root / "attention_distribution_summary.md"
md_path.write_text("\n".join(md_lines) + "\n")

print(f"Wrote {wide_path}")
print(f"Wrote {summary_path}")
print(f"Wrote {md_path}")
if failures:
    print(f"Missing/failed rows: {len(failures)}")
PY

echo "Done. Summary: $OUTPUT_ROOT/attention_distribution_summary.md"
