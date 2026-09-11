#!/usr/bin/env python3
"""
Aggregate per-condition per-subtask accuracy for the Two-Reader Oracle Ablation.

Reads:  <output_dir>/<condition>/<subtask>.json  (produced by run_inference_qwen_oracle.py)
Writes: <output_dir>/two_reader_oracle_table.{csv,md}

Also reports the rate of "Image 3" mentions in <think> blocks per oracle
condition (sanity check that the directive prompt is doing its job).
"""
import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict

CONDITIONS = [
    "none", "T_td_blender", "T_cor", "T_pano", "T_noise",
    "T_pano_gen", "T_cor_gen", "T_td_gen",
    "T_cor_view", "T_cor_view_gen",
]
SUBTASKS = ["anchor", "counting", "relative_distance", "relative_direction"]

# Paired GT↔gen mapping for the "Δ(gen-GT)" comparison.
GEN_PAIRS = [
    ("T_pano", "T_pano_gen"),
    ("T_cor",  "T_cor_gen"),
    ("T_td_blender", "T_td_gen"),
    ("T_cor_view", "T_cor_view_gen"),
]

# Aliases the model may use to refer to the third image
IMAGE_3_RE = re.compile(
    r"\b(image[ _-]?3|third image|the panorama|the top[ -]?down|the dots|the composite|the bird['’]?s[ -]?eye)\b",
    re.IGNORECASE,
)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

DEFAULT_OUTPUT_DIR = "/path/to/scratch/VisualCoT/eval_results/two_reader_oracle"


def load_condition_subtask(output_dir: Path, condition: str, subtask: str):
    p = output_dir / condition / f"{subtask}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def compute_acc(results, subset_sids=None):
    ok = [r for r in results if r.get("status") == "ok"]
    if subset_sids is not None:
        ok = [r for r in ok if r["sample_id"] in subset_sids]
    n = len(ok)
    if n == 0:
        return 0.0, 0, 0
    corr = sum(r.get("accuracy", 0.0) for r in ok)
    return corr / n, int(corr), n


def think_mention_rate(results):
    n = 0
    n_mention = 0
    for r in results:
        if r.get("status") != "ok":
            continue
        text = r.get("final_answer_text") or ""
        m = THINK_RE.search(text)
        think_text = m.group(1) if m else text  # fall back to full output
        n += 1
        if IMAGE_3_RE.search(think_text):
            n_mention += 1
    if n == 0:
        return 0.0, 0, 0
    return n_mention / n, n_mention, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paired", action="store_true",
                        help="Restrict per-condition accuracy to sample_ids "
                             "where ALL 4 conditions produced a valid result. "
                             "Default: use each condition's full set.")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    # Load everything
    runs = {}  # (cond, subtask) -> list of result rows
    for cond in CONDITIONS:
        for st in SUBTASKS:
            payload = load_condition_subtask(out_dir, cond, st)
            if payload is None:
                continue
            runs[(cond, st)] = payload.get("results", [])

    # Skip conditions that have no data files on disk (e.g. T_noise on Gemini)
    active_conditions = [c for c in CONDITIONS if any((c, st) in runs for st in SUBTASKS)]

    # Optional paired-comparison sample_id intersection (per subtask).
    # Restrict to sample_ids that appear with status=ok across ALL active
    # conditions, so every reported accuracy is on the same denominator.
    paired_sids = None
    if args.paired:
        paired_sids = {}
        for st in SUBTASKS:
            sets = []
            for cond in active_conditions:
                rows = runs.get((cond, st), [])
                sets.append({r["sample_id"] for r in rows if r.get("status") == "ok"})
            paired_sids[st] = set.intersection(*sets) if sets else set()

    # Build table
    print(f"\n{'subtask':22}" + "".join(f"{c:>20}" for c in active_conditions))
    print("-" * (22 + 20 * len(active_conditions)))
    rows = []
    overall_correct = defaultdict(int)
    overall_n = defaultdict(int)
    for st in SUBTASKS:
        line = f"{st:22}"
        cond_to_acc = {}
        for cond in active_conditions:
            results = runs.get((cond, st), [])
            sids = paired_sids[st] if args.paired and paired_sids else None
            acc, corr, n = compute_acc(results, sids)
            cond_to_acc[cond] = (acc, corr, n)
            overall_correct[cond] += corr
            overall_n[cond] += n
            line += f"{acc*100:>14.2f}% (n={n:3d})"
        print(line)
        rows.append((st, cond_to_acc))

    # Overall row
    print("-" * (22 + 20 * len(active_conditions)))
    line = f"{'overall':22}"
    overall_acc = {}
    for cond in active_conditions:
        acc = overall_correct[cond] / overall_n[cond] if overall_n[cond] else 0.0
        overall_acc[cond] = (acc, overall_correct[cond], overall_n[cond])
        line += f"{acc*100:>14.2f}% (n={overall_n[cond]:3d})"
    print(line)
    # Restrict later sections to the active set
    CONDITIONS_USED = active_conditions

    # Uplift table
    uplift_conds = [c for c in CONDITIONS_USED if c != "none"]
    print("\nUplift over `none` baseline:")
    print(f"{'subtask':22}" + "".join(f"{c:>20}" for c in uplift_conds))
    print("-" * (22 + 20 * len(uplift_conds)))
    for st, cond_to_acc in rows:
        base = cond_to_acc["none"][0]
        line = f"{st:22}"
        for cond in uplift_conds:
            delta = (cond_to_acc[cond][0] - base) * 100
            sign = "+" if delta >= 0 else ""
            line += f"{sign}{delta:>13.2f} pts     "
        print(line)
    # Overall uplift
    line = f"{'overall':22}"
    for cond in uplift_conds:
        delta = (overall_acc[cond][0] - overall_acc["none"][0]) * 100
        sign = "+" if delta >= 0 else ""
        line += f"{sign}{delta:>13.2f} pts     "
    print("-" * (22 + 20 * len(uplift_conds)))
    print(line)

    # Paired GT vs generated-image comparison: Δ = gen - GT
    pairs = [(gt, gen) for (gt, gen) in GEN_PAIRS if gt in CONDITIONS_USED and gen in CONDITIONS_USED]
    if pairs:
        print("\nPaired: generated-image vs GT (Δ = gen - GT, percentage points):")
        header = f"{'subtask':22}" + "".join(f"  {gt:>10}→{gen:>14}  {'Δ(gen-GT)':>14}" for (gt, gen) in pairs)
        print(header)
        print("-" * len(header))
        for st, cond_to_acc in rows:
            line = f"{st:22}"
            for gt, gen in pairs:
                gt_acc = cond_to_acc[gt][0] * 100
                gen_acc = cond_to_acc[gen][0] * 100
                delta = gen_acc - gt_acc
                sign = "+" if delta >= 0 else ""
                line += f"  {gt_acc:>10.2f}→{gen_acc:>13.2f}%  {sign}{delta:>12.2f}  "
            print(line)
        # Overall
        line = f"{'overall':22}"
        for gt, gen in pairs:
            gt_acc = overall_acc[gt][0] * 100
            gen_acc = overall_acc[gen][0] * 100
            delta = gen_acc - gt_acc
            sign = "+" if delta >= 0 else ""
            line += f"  {gt_acc:>10.2f}→{gen_acc:>13.2f}%  {sign}{delta:>12.2f}  "
        print("-" * len(header))
        print(line)

    # Image-3 mention rate sanity check
    print("\nImage-3 mention rate in <think> blocks (sanity check that the directive prompt worked):")
    for cond in CONDITIONS_USED:
        all_results = []
        for st in SUBTASKS:
            all_results.extend(runs.get((cond, st), []))
        rate, n_m, n_t = think_mention_rate(all_results)
        print(f"  {cond:18}: {rate*100:>6.2f}%  ({n_m}/{n_t} responses)")

    # Save CSV
    out_csv = out_dir / "two_reader_oracle_table.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subtask"] + sum(([f"{c}_acc", f"{c}_n", f"{c}_correct"] for c in CONDITIONS_USED), []))
        for st, cond_to_acc in rows:
            r = [st]
            for cond in CONDITIONS_USED:
                acc, corr, n = cond_to_acc[cond]
                r += [f"{acc:.4f}", n, corr]
            w.writerow(r)
        # overall
        r = ["overall"]
        for cond in CONDITIONS_USED:
            acc, corr, n = overall_acc[cond]
            r += [f"{acc:.4f}", n, corr]
        w.writerow(r)
    print(f"\nWrote {out_csv}")

    # Save Markdown
    out_md = out_dir / "two_reader_oracle_table.md"
    with open(out_md, "w") as f:
        f.write("# Two-Reader Oracle Ablation\n\n")
        f.write("## Accuracy per condition × subtask\n\n")
        f.write("| Subtask |" + " | ".join(CONDITIONS_USED) + " |\n")
        f.write("|---" * (len(CONDITIONS_USED) + 1) + "|\n")
        for st, cond_to_acc in rows:
            f.write(f"| {st} |")
            for cond in CONDITIONS_USED:
                acc, corr, n = cond_to_acc[cond]
                f.write(f" {acc*100:.2f}% ({corr}/{n}) |")
            f.write("\n")
        f.write(f"| **overall** |")
        for cond in CONDITIONS_USED:
            acc, corr, n = overall_acc[cond]
            f.write(f" **{acc*100:.2f}%** ({corr}/{n}) |")

        uplift_conds_md = [c for c in CONDITIONS_USED if c != "none"]
        f.write("\n\n## Uplift over `none` baseline (percentage points)\n\n")
        f.write("| Subtask |" + " | ".join(uplift_conds_md) + " |\n")
        f.write("|---" * (len(uplift_conds_md) + 1) + "|\n")
        for st, cond_to_acc in rows:
            base = cond_to_acc["none"][0]
            f.write(f"| {st} |")
            for cond in uplift_conds_md:
                delta = (cond_to_acc[cond][0] - base) * 100
                sign = "+" if delta >= 0 else ""
                f.write(f" {sign}{delta:.2f} pts |")
            f.write("\n")
        f.write(f"| **overall** |")
        for cond in uplift_conds_md:
            delta = (overall_acc[cond][0] - overall_acc["none"][0]) * 100
            sign = "+" if delta >= 0 else ""
            f.write(f" **{sign}{delta:.2f} pts** |")

        # Paired GT vs generated comparison
        pairs_md = [(gt, gen) for (gt, gen) in GEN_PAIRS if gt in CONDITIONS_USED and gen in CONDITIONS_USED]
        if pairs_md:
            f.write("\n\n## Paired: generated-image vs GT (Δ = gen − GT)\n\n")
            f.write("| Subtask |" + " | ".join(f"{gt} → {gen} (Δ)" for (gt, gen) in pairs_md) + " |\n")
            f.write("|---" * (len(pairs_md) + 1) + "|\n")
            for st, cond_to_acc in rows:
                f.write(f"| {st} |")
                for gt, gen in pairs_md:
                    gt_acc = cond_to_acc[gt][0] * 100
                    gen_acc = cond_to_acc[gen][0] * 100
                    delta = gen_acc - gt_acc
                    sign = "+" if delta >= 0 else ""
                    f.write(f" {gt_acc:.2f}% → {gen_acc:.2f}% ({sign}{delta:.2f}) |")
                f.write("\n")
            f.write(f"| **overall** |")
            for gt, gen in pairs_md:
                gt_acc = overall_acc[gt][0] * 100
                gen_acc = overall_acc[gen][0] * 100
                delta = gen_acc - gt_acc
                sign = "+" if delta >= 0 else ""
                f.write(f" **{gt_acc:.2f}% → {gen_acc:.2f}% ({sign}{delta:.2f})** |")

        f.write("\n\n## Image-3 mention rate\n\n")
        f.write("| Condition | Mention rate |\n|---|---|\n")
        for cond in CONDITIONS_USED:
            all_results = []
            for st in SUBTASKS:
                all_results.extend(runs.get((cond, st), []))
            rate, n_m, n_t = think_mention_rate(all_results)
            f.write(f"| {cond} | {rate*100:.2f}% ({n_m}/{n_t}) |\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
