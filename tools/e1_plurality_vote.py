#!/usr/bin/env python
"""E1a — plurality-vote ensemble baseline over the view experts (ADAPTIVE_SELECTION.md).

Realizable-without-routing reference: per question, take the most common predicted
answer across the available experts; ties break toward the expert with the higher
accuracy on that benchmark. Compared against best-single and the E0 oracle.

Also dumps a per-question router task file (question text + expert answers +
correctness) used by E1b/E1c.
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e0_per_question_oracle as e0

COSMIC = set(e0.COSMIC)


def answer_of(bench, meta):
    if bench == "mmsi":
        return meta.get("ExtractedAnswer")
    return meta.get("predicted_answer")


def gt_of(bench, meta):
    if bench in COSMIC:
        idx = meta.get("correct_answer_idx")
        return chr(65 + int(idx)) if idx not in (None, "") else None
    if bench in ("mindcube", "blink"):
        return meta.get("gt_answer")
    if bench == "mmsi":
        return meta.get("GroundTruth")
    return meta.get("correct_answer")


def norm(a):
    return str(a).strip().upper() if a not in (None, "") else None


def resolve_sources():
    """Same resolution the E0 script does at runtime."""
    nt_tag = None
    for tag in (e0.NT1, e0.NT2):
        a = e0.acc(e0.load_spatial(f"{e0.V}/{tag}_mcqs_anchor_normalized", "sample_id"))
        if abs(a - e0.EXPECTED["no_think"]["anchor"]) < 0.15:
            nt_tag = tag
            break
    assert nt_tag
    sources = dict(e0.SOURCES)
    sources["no_think"] = e0.nt_sources(nt_tag)
    for d in (f"{e0.V}/eval_outputs/{e0.TD}_mmsi",
              f"{e0.V}/eval_outputs/BAGEL_format_corner_view_visual_only_bridge_masked_mmsi"):
        if os.path.exists(os.path.join(d, "evaluated_model_results_run_1.json")) and \
                abs(e0.acc(e0.load_mmsi(d)) - e0.EXPECTED["td"]["mmsi"]) < 0.15:
            sources["td"]["mmsi"]["dir"] = d
            break
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default=f"{e0.V}/eval_results/adaptive_selection_e0")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sources = resolve_sources()
    data = {}
    for expert, benches in sources.items():
        for bench, spec in benches.items():
            data.setdefault(bench, {})[expert] = e0.load(spec)

    rows = []
    for bench in sorted(data):
        experts = data[bench]
        # priority: higher benchmark accuracy first (tie-break order for the vote)
        prio = sorted(experts, key=lambda e: -e0.acc(experts[e]))
        common = set.intersection(*(set(experts[e]) for e in experts))
        n = len(common)
        vote_correct = 0
        task_rows = []
        for s in sorted(common):
            answers = {e: norm(answer_of(bench, experts[e][s]["meta"])) for e in prio}
            votes = Counter(a for a in answers.values() if a is not None)
            if votes:
                top = max(votes.values())
                tied = [a for a, c in votes.items() if c == top]
                # tie-break: first expert (in priority order) whose answer is tied
                voted = next(answers[e] for e in prio if answers[e] in tied)
            else:
                voted = None
            # correctness of the voted answer: reuse an agreeing expert's official
            # correctness (matters for MMSI where the judge overrides letter equality)
            agree = next((e for e in prio if answers[e] == voted and voted is not None), None)
            if agree is not None:
                correct = experts[agree][s]["correct"]
            else:
                correct = norm(gt_of(bench, experts[prio[0]][s]["meta"])) == voted
            vote_correct += bool(correct)
            task_rows.append({
                "sample_id": s,
                "question": (experts[prio[0]][s]["meta"].get("question")
                             or experts[prio[0]][s]["meta"].get("Question")),
                "answers": answers,
                "correct_by_expert": {e: experts[e][s]["correct"] for e in prio},
                "vote": voted, "vote_correct": bool(correct),
            })
        single = {e: 100.0 * sum(experts[e][s]["correct"] for s in common) / n for e in experts}
        oracle = 100.0 * sum(any(experts[e][s]["correct"] for e in experts) for s in common) / n
        rows.append({"bench": bench, "n": n, "experts": prio, "single": single,
                     "best_single": max(single.values()), "vote": 100.0 * vote_correct / n,
                     "oracle": oracle})
        with open(os.path.join(args.output_dir, f"router_task_{bench}.json"), "w") as f:
            json.dump(task_rows, f)

    with open(os.path.join(args.output_dir, "e1_vote_report.json"), "w") as f:
        json.dump(rows, f, indent=1)

    print(f"{'bench':<20}{'n':>6}{'#exp':>5}{'best-single':>12}{'vote':>8}{'oracle':>8}")
    for r in rows:
        print(f"{r['bench']:<20}{r['n']:>6}{len(r['experts']):>5}"
              f"{r['best_single']:>12.2f}{r['vote']:>8.2f}{r['oracle']:>8.2f}")


if __name__ == "__main__":
    main()
