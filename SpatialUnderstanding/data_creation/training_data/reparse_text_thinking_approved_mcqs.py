#!/usr/bin/env python3
"""
Re-parse the existing T_cot JSONLs with the fixed parse_response (handles
no-tag <think> output) and identify rows that still need fresh inference
(those without a saved 'raw' field).

For rows where status was 'parse_fail' or 'wrong' but `raw` was saved, we
re-run the parser; for `missing_gold` (no inference attempt was made),
we list the sample_ids so the annotator can be re-launched on just those.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotate_text_reasoning_approved_mcqs import parse_response

ROOT = Path(
    "/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness/text_thinking_approved_mcqs"
)


def reparse_subtask(subtask):
    p = ROOT / f"{subtask}.jsonl"
    rows = [json.loads(ln) for ln in open(p)]

    new_rows = []
    needs_inference = []
    counters = {"ok": 0, "wrong": 0, "parse_fail": 0, "missing_gold_recovered": 0, "still_missing": 0, "no_raw": 0}

    for r in rows:
        old_status = r.get("status")
        raw = r.get("raw")
        gold = r.get("gold_answer")

        if raw and gold:
            think_text, pred, status = parse_response(raw, gold)
            new_r = {
                "sample_id": r["sample_id"],
                "subtask": subtask,
                "think_text": think_text,
                "predicted_answer": pred,
                "gold_answer": gold,
                "status": status,
                "raw": raw if status != "ok" else None,
            }
            counters[status] = counters.get(status, 0) + 1
            new_rows.append(new_r)
        elif old_status == "missing_gold":
            # These rows were skipped by the buggy build_messages — needs re-inference.
            needs_inference.append(r["sample_id"])
            new_rows.append(r)
            counters["still_missing"] += 1
        else:
            new_rows.append(r)
            counters["no_raw"] += 1

    out_p = p.with_suffix(".reparsed.jsonl")
    with open(out_p, "w") as f:
        for r in new_rows:
            f.write(json.dumps(r) + "\n")
    print(f"{subtask}: {counters}")
    print(f"  -> {out_p}")
    if needs_inference:
        miss_p = ROOT / f"{subtask}.needs_inference.json"
        json.dump(needs_inference, open(miss_p, "w"), indent=2)
        print(f"  needs inference for {len(needs_inference)} samples -> {miss_p}")
    return counters, needs_inference


def main():
    total = {"ok": 0, "wrong": 0, "parse_fail": 0, "still_missing": 0, "no_raw": 0}
    all_missing = {}
    for subtask in ["anchor", "counting", "relative_distance", "relative_direction"]:
        c, miss = reparse_subtask(subtask)
        for k, v in c.items():
            total[k] = total.get(k, 0) + v
        if miss:
            all_missing[subtask] = miss
    print()
    print(f"TOTAL: {total}")


if __name__ == "__main__":
    main()
