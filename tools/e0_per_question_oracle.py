#!/usr/bin/env python
"""E0 — per-question oracle over the four view experts (ADAPTIVE_SELECTION.md).

Joins per-sample eval results of the four experts (panorama+VDrop, top-down
corner-view+VDrop, point-matching+VDrop, No-Think) by sample id, per benchmark,
and computes:
  * each expert's accuracy (sanity-checked against the paper/PROGRESS numbers),
  * the union-correctness oracle (any expert correct),
  * complementarity: only-expert-X-correct counts, pairwise a-correct-b-wrong,
  * the per-question route-label distribution (which experts got it right).

Read-only over the eval dirs; writes JSON + a markdown report to --output_dir.
"""

import argparse
import glob
import json
import os
from collections import Counter

V = "/path/to/scratch/VisualCoT"
S = "/path/to/scratch"

PANO = "BAGEL_format_mix_all_balance_visual_only_bridge_masked_lora_7k"
PANO6K = "BAGEL_format_mix_all_balance_visual_only_bridge_masked_lora_6k"  # dir name typo, holds 7k results
TD = "BAGEL_format_corner_view_visual_only_bridge_masked_lora_7k"
PM = "BAGEL_format_pm_no_rotation_visual_only_bridge_masked_lora_7k"
NT1 = "BAGEL_format_mix_all_balance_no_think_lora_10k"
NT2 = "BAGEL_format_mix_anchor_counting_distance_balance_no_think_lora_lora_10k"

COSMIC = ["anchor", "counting", "relative_distance", "relative_direction"]

# (expert, benchmark) -> dict(kind=..., path=...)
# kind: "spatial" (merged/shard JSON with results[].accuracy) or "mmsi" (judged list)
def cosmic(tag, sub):
    return {"kind": "spatial", "dir": f"{V}/{tag}_mcqs_{sub}_normalized", "id": "sample_id"}

SOURCES = {
    "pano": {
        **{sub: cosmic(PANO, sub) for sub in COSMIC},
        "mmsi": {"kind": "mmsi", "dir": f"{V}/eval_outputs/{PANO6K}_mmsi"},
        "mindcube": {"kind": "spatial", "dir": f"{V}/eval_outputs/{PANO6K}_mindcube", "id": "sample_id"},
        "blink": {"kind": "spatial", "dir": f"{V}/eval_outputs/{PANO6K}_blink", "id": "sample_id"},
        # omnispatial / stare: results only exist off-cluster (EXPERIMENTS.md) — absent here
    },
    "td": {
        **{sub: cosmic(TD, sub) for sub in COSMIC},
        "mmsi": {"kind": "mmsi", "dir": f"{V}/eval_outputs/{TD}_mmsi"},  # run picked by --td_mmsi_dir check
        "mindcube": {"kind": "spatial", "dir": f"{S}/mindcube/{TD}", "id": "sample_id"},
        "blink": {"kind": "spatial", "dir": f"{S}/blink/{TD}", "id": "sample_id"},
        "omnispatial": {"kind": "spatial", "dir": f"{V}/omnispatial/{TD}_complex_logic_perspective_taking", "id": "omni"},
        "stare": {"kind": "spatial", "dir": f"{V}/stare_perspective/{TD}", "id": "qid"},
    },
    "pm": {
        **{sub: cosmic(PM, sub) for sub in COSMIC},
        "mmsi": {"kind": "mmsi", "dir": f"{V}/eval_outputs/{PM}_mmsi"},
        # mindcube / blink / omnispatial / stare: off-cluster only
    },
    # no_think tag resolved at runtime between NT1/NT2 (--nt_tag to force)
}

def nt_sources(tag):
    return {
        **{sub: cosmic(tag, sub) for sub in COSMIC},
        "mmsi": {"kind": "mmsi", "dir": f"{V}/{tag}_mmsi_eval"},
        "mindcube": {"kind": "spatial", "dir": f"{S}/mindcube/{tag}", "id": "sample_id"},
        "blink": {"kind": "spatial", "dir": f"{S}/blink/{tag}", "id": "sample_id"},
        "omnispatial": {"kind": "spatial", "dir": f"{V}/omnispatial/{tag}_complex_logic_perspective_taking", "id": "omni"},
        "stare": {"kind": "spatial", "dir": f"{V}/stare_perspective/{tag}", "id": "qid"},
    }

# Expected accuracies (paper Table 1 / PROGRESS Setting A) for sanity checks.
EXPECTED = {
    "pano": {"anchor": 89.2, "counting": 78.8, "relative_distance": 74.8, "relative_direction": 93.2,
             "mmsi": 26.0, "mindcube": 34.09, "blink": 62.41},
    "td":   {"anchor": 94.4, "counting": 76.8, "relative_distance": 72.0, "relative_direction": 89.2,
             "mmsi": 32.0, "mindcube": 36.48, "blink": 57.14, "stare": 26.0},
    "pm":   {"anchor": 92.8, "counting": 78.8, "relative_distance": 78.0, "relative_direction": 91.6,
             "mmsi": 28.3},
    "no_think": {"anchor": 86.8, "counting": 82.4, "relative_distance": 67.6, "relative_direction": 85.6,
                 "mmsi": 27.4, "mindcube": 41.1, "stare": 24.8, "blink": 45.1},
}
# td COSMIC dirs may hold run 1 (anchor 93.6 / counting 83.2 / dist 68.4 / dir 92.0 or 91.6);
# omni expectations are checked per split (CL/PT) inside the report instead.
TD_RUN1_COSMIC = {"anchor": 93.6, "counting": 76.8, "relative_distance": 72.0, "relative_direction": 89.2}


def load_spatial(dirpath, id_field):
    """Merged-if-present else shards; dedup by id (first seen, matching merge scripts)."""
    merged = os.path.join(dirpath, "inference_results_bagel_merged.json")
    if not os.path.exists(merged):
        merged = os.path.join(dirpath, "merged_results.json")
    if os.path.exists(merged):
        files = [merged]
    else:
        files = sorted(glob.glob(os.path.join(dirpath, "inference_results*shard*.json"))) or \
                sorted(glob.glob(os.path.join(dirpath, "inference_results*.json")))
    out = {}
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        recs = d["results"] if isinstance(d, dict) else d
        for r in recs:
            if id_field == "omni":  # benchmark row id restarts per task file -> qualify it
                sid = f"{r['task_type']}|{r['id']}"
            else:
                sid = str(r[id_field])
            if sid not in out:
                out[sid] = {"correct": float(r["accuracy"]) > 0.5, "meta": r}
    return out


def load_mmsi(dirpath):
    fp = os.path.join(dirpath, "evaluated_model_results_run_1.json")
    with open(fp) as f:
        recs = json.load(f)
    out = {}
    for r in recs:
        sid = str(r["Id"])
        if sid not in out:
            out[sid] = {"correct": str(r.get("LLMJudgeResult")) == "True", "meta": r}
    return out


def load(spec):
    if spec["kind"] == "mmsi":
        return load_mmsi(spec["dir"])
    return load_spatial(spec["dir"], spec["id"])


def acc(d):
    return 100.0 * sum(v["correct"] for v in d.values()) / len(d) if d else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default=f"{V}/eval_results/adaptive_selection_e0")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- resolve the no_think tag by COSMIC accuracy match
    nt_tag, nt_reason = None, None
    for tag in (NT1, NT2):
        try:
            a = acc(load_spatial(f"{V}/{tag}_mcqs_anchor_normalized", "sample_id"))
        except Exception:
            continue
        if abs(a - EXPECTED["no_think"]["anchor"]) < 0.15:
            nt_tag, nt_reason = tag, f"anchor={a:.1f} matches expected {EXPECTED['no_think']['anchor']}"
            break
    assert nt_tag, "neither no_think candidate matches the expected anchor accuracy"
    SOURCES["no_think"] = nt_sources(nt_tag)

    # --- resolve the td MMSI run-2 dir
    td_mmsi_cands = [f"{V}/eval_outputs/{TD}_mmsi",
                     f"{V}/eval_outputs/BAGEL_format_corner_view_visual_only_bridge_masked_mmsi"]
    td_mmsi_pick, td_mmsi_note = None, []
    for d in td_mmsi_cands:
        if not os.path.exists(os.path.join(d, "evaluated_model_results_run_1.json")):
            td_mmsi_note.append(f"{d}: no judged file")
            continue
        a = acc(load_mmsi(d))
        td_mmsi_note.append(f"{d}: {a:.2f}")
        if abs(a - EXPECTED["td"]["mmsi"]) < 0.15:
            td_mmsi_pick = d
    if td_mmsi_pick:
        SOURCES["td"]["mmsi"]["dir"] = td_mmsi_pick

    # --- load everything
    data = {}   # bench -> expert -> {sid: {correct, meta}}
    checks = []
    for expert, benches in SOURCES.items():
        for bench, spec in benches.items():
            try:
                d = load(spec)
            except Exception as e:
                checks.append((expert, bench, "LOAD_FAIL", str(e)))
                continue
            a = acc(d)
            exp = EXPECTED.get(expert, {}).get(bench)
            status = "ok" if exp is None else ("MATCH" if abs(a - exp) < 0.15 else f"MISMATCH(exp {exp})")
            if expert == "td" and bench in TD_RUN1_COSMIC and status.startswith("MISMATCH") \
                    and abs(a - TD_RUN1_COSMIC[bench]) < 0.15:
                status = "MATCH_RUN1"
            checks.append((expert, bench, f"{a:.2f} n={len(d)}", status))
            data.setdefault(bench, {})[expert] = d

    # --- oracle + complementarity per benchmark
    report = {"no_think_tag": nt_tag, "no_think_reason": nt_reason,
              "td_mmsi_note": td_mmsi_note, "checks": [list(c) for c in checks],
              "benchmarks": {}}
    for bench, experts in sorted(data.items()):
        names = sorted(experts)
        common = set.intersection(*(set(experts[e]) for e in names))
        n = len(common)
        single = {e: 100.0 * sum(experts[e][s]["correct"] for s in common) / n for e in names}
        oracle = 100.0 * sum(any(experts[e][s]["correct"] for e in names) for s in common) / n
        only = {e: sum(experts[e][s]["correct"] and
                       not any(experts[o][s]["correct"] for o in names if o != e)
                       for s in common) for e in names}
        none_right = sum(not any(experts[e][s]["correct"] for e in names) for s in common)
        pair = {f"{a}>{b}": sum(experts[a][s]["correct"] and not experts[b][s]["correct"] for s in common)
                for a in names for b in names if a != b}
        n_correct_hist = Counter(sum(experts[e][s]["correct"] for e in names) for s in common)
        # per-question route labels (winning experts), for E2
        labels = {s: [e for e in names if experts[e][s]["correct"]] for s in common}
        report["benchmarks"][bench] = {
            "experts": names, "n_common": n, "single": single, "oracle": oracle,
            "best_single": max(single, key=single.get),
            "gain_over_best": oracle - max(single.values()),
            "only_counts": only, "none_correct": none_right,
            "pairwise_a_correct_b_wrong": pair,
            "n_experts_correct_hist": dict(sorted(n_correct_hist.items())),
        }
        with open(os.path.join(args.output_dir, f"route_labels_{bench}.json"), "w") as f:
            json.dump(labels, f)

    # omnispatial split accuracies (CL / PT) for the OOD-average convention
    if "omnispatial" in data:
        omni = {}
        for e, d in data["omnispatial"].items():
            for split in ("Complex_Logic", "Perspective_Taking"):
                sub = [v for v in d.values() if v["meta"].get("task_type") == split]
                omni.setdefault(e, {})[split] = 100.0 * sum(x["correct"] for x in sub) / len(sub)
        # oracle per split
        names = sorted(data["omnispatial"])
        common = set.intersection(*(set(data["omnispatial"][e]) for e in names))
        for split in ("Complex_Logic", "Perspective_Taking"):
            ss = [s for s in common
                  if data["omnispatial"][names[0]][s]["meta"].get("task_type") == split]
            omni.setdefault("oracle", {})[split] = \
                100.0 * sum(any(data["omnispatial"][e][s]["correct"] for e in names) for s in ss) / len(ss)
        report["omnispatial_splits"] = omni

    out = os.path.join(args.output_dir, "e0_oracle_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)

    # --- console summary
    print(f"no_think tag: {nt_tag} ({nt_reason})")
    for line in td_mmsi_note:
        print("td mmsi:", line)
    print("\n== sanity checks ==")
    for c in checks:
        print("  ", *c)
    print("\n== oracle ==")
    hdr = f"{'bench':<20}{'n':>6}" + "".join(f"{e:>10}" for e in ["no_think", "pano", "pm", "td"]) + f"{'oracle':>9}{'gain':>7}  none/only"
    print(hdr)
    for bench, r in report["benchmarks"].items():
        cells = "".join(f"{r['single'].get(e, float('nan')):>10.2f}" if e in r["single"] else f"{'—':>10}"
                        for e in ["no_think", "pano", "pm", "td"])
        print(f"{bench:<20}{r['n_common']:>6}{cells}{r['oracle']:>9.2f}{r['gain_over_best']:>7.2f}"
              f"  none={r['none_correct']} only={r['only_counts']}")
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
