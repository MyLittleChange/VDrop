#!/usr/bin/env python3
"""Audit duplicate sample_ids in mix_all text_thinking annotated files.

Renders an HTML report so the user can confirm that:
  - duplicate ids are real variants (different question text), not annotation errors
  - the (sample_id, question_text) lookup key resolves each variant cleanly to
    a unique visual_only source sample

Run:
    python audit_text_thinking_dups.py
Output:
    /path/to/scratch/infinigen/training_data_mix_all_balance/audit_text_thinking_dups.html
"""

import argparse
import html
import json
import os
from collections import defaultdict


TEXT_THINKING_DIR = "/path/to/scratch/infinigen/training_data_mix_all_balance/text_thinking"
CATEGORIES = ["counting", "anchor", "relative_distance", "spatial", "perspective_taking"]

VISUAL_ONLY_SOURCES = {
    "counting": [
        "/path/to/scratch/infinigen/outputs_rendered/dataset_counting_questions_filtered_V4.json",
        "/path/to/scratch/infinigen/spatial/dataset_counting_questions_filtered_V5_normalized.json",
    ],
    "anchor": [
        "/path/to/scratch/infinigen/dataset_anchor_questions_filtered_V4_normalized.json",
        "/path/to/scratch/infinigen/spatial/dataset_anchor_questions_filtered_V5_normalized.json",
    ],
    "relative_distance": [
        "/path/to/scratch/infinigen/outputs_rendered/dataset_relative_distance_questions_filtered_V4.json",
        "/path/to/scratch/infinigen/spatial/dataset_relative_distance_questions_filtered_V5.json",
    ],
    "spatial": [
        "/path/to/scratch/infinigen/dataset_spatial_questions_filtered_V4_normalized.json",
        "/path/to/scratch/infinigen/spatial/dataset_spatial_questions_filtered_V5_normalized.json",
    ],
    "perspective_taking": [
        "/path/to/scratch/infinigen/dataset_perspective_taking_questions_filtered_V4_normalized.json",
        "/path/to/scratch/infinigen/spatial/dataset_perspective_taking_questions_filtered_V5_normalized.json",
    ],
}

DEFAULT_OUT = "/path/to/scratch/infinigen/training_data_mix_all_balance/audit_text_thinking_dups.html"


def extract_question(human_val: str) -> str:
    last = human_val.rfind("<image>")
    tail = human_val[last + len("<image>"):] if last != -1 else human_val
    return tail.lstrip("\n").strip()


def extract_think_and_answer(gpt_val: str):
    think = ""
    answer = ""
    ts = gpt_val.find("<think>")
    te = gpt_val.find("</think>")
    if ts != -1 and te != -1:
        think = gpt_val[ts + len("<think>"):te].strip()
    aas = gpt_val.find("<answer>")
    aae = gpt_val.find("</answer>")
    if aas != -1 and aae != -1:
        answer = gpt_val[aas + len("<answer>"):aae].strip()
    return think, answer


def load_visual_only_index(category: str):
    """Returns {sample_id: [{src_file, scene_id, question_both_views, question_type, options, gt_idx}, ...]}."""
    idx = defaultdict(list)
    for fp in VISUAL_ONLY_SOURCES[category]:
        if not os.path.exists(fp):
            print(f"  [warn] missing source {fp}")
            continue
        with open(fp) as f:
            data = json.load(f)
        src_tag = "V4" if "V4" in fp else ("V5" if "V5" in fp else os.path.basename(fp))
        for s in data:
            sid = s.get("sample_id")
            if not sid:
                continue
            opts_1 = s.get("options_user_1")
            opts_2 = s.get("options_user_2")
            if opts_1 is not None:
                options = opts_1
                gt_idx = s.get("user_1_gt_answer_idx")
            elif opts_2 is not None:
                options = opts_2
                gt_idx = s.get("user_2_gt_answer_idx")
            else:
                options = s.get("options", [])
                gt_idx = s.get("correct_answer_idx")
            idx[sid].append({
                "src":  src_tag,
                "scene_id": s.get("scene_id", ""),
                "question_type": s.get("question_type", ""),
                "question_both_views": (s.get("question_both_views", "") or "").strip(),
                "options": options,
                "gt_idx": gt_idx,
            })
    return idx


def render_options(options, gt_idx):
    if not options:
        return ""
    parts = []
    for i, opt in enumerate(options):
        marker = " ←" if (gt_idx is not None and i == gt_idx) else ""
        parts.append(f"{chr(65 + i)}) {html.escape(str(opt))}{marker}")
    return "<br>".join(parts)


def question_matches_visual_only(annotated_q: str, vo_q: str) -> str:
    """Returns 'exact', 'prefix', or 'no_match'."""
    a = annotated_q.strip()
    b = vo_q.strip()
    if a == b:
        return "exact"
    if a.startswith(b) or b.startswith(a):
        return "prefix"
    return "no_match"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_thinking_dir", default=TEXT_THINKING_DIR)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--max_groups_per_category", type=int, default=0,
                        help="Cap dup-id groups rendered per category (0 = all)")
    args = parser.parse_args()

    sections = []
    summary_rows = []
    total_dups = 0
    total_resolved = 0
    total_unresolved = 0

    for cat in CATEGORIES:
        path = os.path.join(args.text_thinking_dir, f"{cat}.jsonl")
        if not os.path.exists(path):
            print(f"  [skip] {path}")
            continue
        rows_by_id = defaultdict(list)
        with open(path) as f:
            for ln in f:
                d = json.loads(ln)
                rows_by_id[d.get("id")].append(d)
        dup_groups = [(sid, rs) for sid, rs in rows_by_id.items() if len(rs) > 1]
        if not dup_groups:
            print(f"  {cat}: no dup ids")
            continue
        if args.max_groups_per_category > 0:
            dup_groups = dup_groups[:args.max_groups_per_category]

        vo_index = load_visual_only_index(cat)

        section_html = [f'<h2>{cat} <small>({len(dup_groups)} dup-id groups)</small></h2>']
        cat_resolved = 0
        cat_unresolved = 0

        for sid, rows in dup_groups:
            vo_samples = vo_index.get(sid, [])

            section_html.append(
                f'<details open><summary><code>{html.escape(sid)}</code> '
                f'— {len(rows)} annotated rows / {len(vo_samples)} visual_only samples</summary>'
            )
            section_html.append('<table class="dup">')
            section_html.append(
                "<tr><th>annotated row</th><th>annotated question</th>"
                "<th>think</th><th>answer</th>"
                "<th>matched visual_only sample</th><th>match</th></tr>"
            )

            for ridx, row in enumerate(rows):
                convs = row.get("conversations", [])
                human = next((c["value"] for c in convs if c.get("from") == "human"), "")
                gpt   = next((c["value"] for c in convs if c.get("from") == "gpt"),   "")
                aq = extract_question(human)
                think, answer = extract_think_and_answer(gpt)

                # find best vo match
                best = None
                best_kind = "no_match"
                for vo in vo_samples:
                    kind = question_matches_visual_only(aq, vo["question_both_views"])
                    if kind == "exact":
                        best, best_kind = vo, "exact"
                        break
                    if kind == "prefix" and best_kind != "exact":
                        best, best_kind = vo, "prefix"

                if best_kind == "exact":
                    badge = '<span class="ok">✓ exact</span>'
                    cat_resolved += 1
                elif best_kind == "prefix":
                    badge = '<span class="warn">~ prefix</span>'
                    cat_resolved += 1
                else:
                    badge = '<span class="bad">✗ no match</span>'
                    cat_unresolved += 1

                vo_html = ""
                if best is not None:
                    vo_html = (
                        f'<b>src:</b> {best["src"]}<br>'
                        f'<b>scene:</b> {html.escape(best["scene_id"])}<br>'
                        f'<b>type:</b> {html.escape(best["question_type"])}<br>'
                        f'<b>q:</b> {html.escape(best["question_both_views"])}<br>'
                        f'<b>opts:</b><br>{render_options(best["options"], best["gt_idx"])}'
                    )
                else:
                    vo_html = (
                        '<i>no candidate found</i><br>'
                        f'<small>{len(vo_samples)} vo samples for this id</small>'
                    )

                section_html.append(
                    f"<tr>"
                    f"<td>{ridx}</td>"
                    f'<td class="q">{html.escape(aq)}</td>'
                    f'<td class="think">{html.escape(think)}</td>'
                    f"<td>{html.escape(answer)}</td>"
                    f'<td class="vo">{vo_html}</td>'
                    f"<td>{badge}</td>"
                    f"</tr>"
                )

            section_html.append("</table></details>")

        sections.append("\n".join(section_html))
        summary_rows.append((cat, len(dup_groups), cat_resolved, cat_unresolved))
        total_dups += len(dup_groups)
        total_resolved += cat_resolved
        total_unresolved += cat_unresolved

    summary_html = ['<h2>Summary</h2>',
                    '<table class="summary"><tr><th>category</th><th>dup-id groups</th>'
                    '<th>annotated rows resolved</th><th>annotated rows unresolved</th></tr>']
    for cat, ngroups, ok, bad in summary_rows:
        summary_html.append(
            f'<tr><td>{cat}</td><td>{ngroups}</td>'
            f'<td class="ok-bg">{ok}</td>'
            f'<td class="{"bad-bg" if bad else ""}">{bad}</td></tr>'
        )
    summary_html.append(
        f'<tr class="total"><td>TOTAL</td><td>{total_dups}</td>'
        f'<td>{total_resolved}</td><td>{total_unresolved}</td></tr>'
    )
    summary_html.append("</table>")

    css = """
    body { font-family: -apple-system, sans-serif; margin: 24px; max-width: 1600px; }
    h2 { border-bottom: 2px solid #333; padding-bottom: 4px; margin-top: 36px; }
    h2 small { color: #666; font-weight: normal; }
    table { border-collapse: collapse; margin: 12px 0; width: 100%; }
    th, td { border: 1px solid #ccc; padding: 6px 10px; vertical-align: top; font-size: 13px; }
    th { background: #f0f0f0; }
    table.dup td.q { width: 22%; }
    table.dup td.think { width: 30%; font-size: 12px; color: #333; }
    table.dup td.vo { width: 28%; font-size: 12px; }
    table.summary { width: auto; }
    .ok { color: #0a7a2f; font-weight: bold; }
    .warn { color: #b07a00; font-weight: bold; }
    .bad { color: #b00020; font-weight: bold; }
    .ok-bg { background: #e8f5e9; }
    .bad-bg { background: #fde7ec; }
    .total td { font-weight: bold; background: #f7f7f7; }
    details { margin: 8px 0; }
    summary { cursor: pointer; padding: 6px; background: #fafafa; border: 1px solid #eee; }
    code { background: #fff3cd; padding: 1px 4px; border-radius: 3px; }
    """

    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Text-thinking dup-id audit</title>"
        f"<style>{css}</style></head><body>"
        "<h1>Mix_all text_thinking dup-id audit</h1>"
        "<p>For each duplicate <code>sample_id</code> in the annotated mix_all category files, "
        "this report shows every annotated row and the visual_only source sample that the "
        "<code>(sample_id, question)</code> lookup would resolve to. "
        "<span class='ok'>✓ exact</span> = annotated question equals visual_only question. "
        "<span class='warn'>~ prefix</span> = one is a prefix of the other (the lookup's prefix "
        "fallback handles this). <span class='bad'>✗ no match</span> = unresolvable; lookup will fail.</p>"
        + "\n".join(summary_html)
        + "\n".join(sections)
        + "</body></html>"
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(page)
    print(f"\nWrote {args.output}")
    print(f"Total dup-id groups: {total_dups}")
    print(f"Annotated rows resolved (exact|prefix): {total_resolved}")
    print(f"Annotated rows unresolved: {total_unresolved}")


if __name__ == "__main__":
    main()
