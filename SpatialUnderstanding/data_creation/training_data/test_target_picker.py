#!/usr/bin/env python3
"""Smoke test for find_target_object: load real spatial / perspective_taking
rows where the question contains a color qualifier and the scene has multiple
instances of question_object, then check the picker's choice.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from text_reasoning_prompts.common import (
    find_objects_by_category,
    find_target_object,
    _COLOR_WORDS,
)


SCRATCH = "/path/to/scratch"
INFINIGEN = f"{SCRATCH}/infinigen"
ANKUR = "/path/to/scratch/spatial_collab_dataset/scenes"


def load_meta_v5_spatial():
    p = f"{INFINIGEN}/spatial/dataset_spatial_questions_filtered_V5_normalized.json"
    return {r["sample_id"]: r for r in json.load(open(p))}


def load_meta_v4_spatial():
    p = f"{INFINIGEN}/dataset_spatial_questions_filtered_V4_normalized.json"
    return {r["sample_id"]: r for r in json.load(open(p))}


def load_visible(scene_root: str) -> dict:
    p = os.path.join(scene_root, "visible_objects_with_descriptions.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p))


def scene_root_v5(meta_row: dict) -> str:
    return f"{INFINIGEN}/spatial/{meta_row['room_part']}/{meta_row['scene_id']}"


def scene_root_v4(meta_row: dict) -> str:
    return f"{INFINIGEN}/outputs_rendered/{meta_row['room_part']}/{meta_row['scene_id']}"


def find_multi_instance_rows(meta_by_id: dict, scene_root_fn, max_n: int = 30):
    """Return rows where the scene has ≥2 instances of question_object visible
    in at least one camera (so the picker actually has a decision to make)."""
    out = []
    for sid, m in meta_by_id.items():
        obj = m.get("question_object")
        if not obj:
            continue
        q = m.get("user_1_question") or m.get("user_2_question") or ""
        # Need a color word in the question for rule 1 to fire.
        q_low = q.lower()
        has_color = any(c in q_low for c in _COLOR_WORDS)
        if not has_color:
            continue
        try:
            visible = load_visible(scene_root_fn(m))
        except Exception:
            continue
        if not visible:
            continue
        for cam_key in ("camera_0_0", "camera_1_0"):
            hits = find_objects_by_category(visible, cam_key, obj)
            if len(hits) >= 2:
                out.append((sid, m, q, cam_key, hits))
                break
        if len(out) >= max_n:
            break
    return out


def main():
    cases = []
    for label, loader, root_fn in [
        ("v5", load_meta_v5_spatial, scene_root_v5),
        ("v4", load_meta_v4_spatial, scene_root_v4),
    ]:
        try:
            meta_by_id = loader()
        except Exception as e:
            print(f"[skip {label}] could not load metadata: {e}")
            continue
        rows = find_multi_instance_rows(meta_by_id, root_fn, max_n=10)
        print(f"== {label} spatial: found {len(rows)} multi-instance + color-qualified rows ==")
        for sid, m, q, cam, hits in rows:
            cases.append((label, sid, m, q, cam, hits))

    print()
    print(f"Total cases: {len(cases)}")
    print("=" * 80)

    n_resolved = 0
    n_ambiguous = 0
    n_rule_color = 0
    n_rule_desc = 0
    for label, sid, m, q, cam, hits in cases[:20]:
        print(f"[{label}] {sid}  question_object={m['question_object']!r}  ({cam})")
        print(f"  Q: {q}")
        print(f"  candidates ({len(hits)}):")
        for oid, obj in hits:
            print(f"    - {obj.get('name')}  color={obj.get('color')!r}  desc={obj.get('description')!r}")
        pick = find_target_object(hits, q)
        if pick["ambiguous"]:
            n_ambiguous += 1
            print(f"  → AMBIGUOUS ({pick['reason']})")
        else:
            n_resolved += 1
            chosen = pick["chosen"]
            print(f"  → CHOSE {chosen.get('name')!r}  color={chosen.get('color')!r}  ({pick['reason']})")
            if "color" in pick["reason"]:
                n_rule_color += 1
            elif "description" in pick["reason"]:
                n_rule_desc += 1
            # Sanity: chosen color or description tokens must overlap question
            q_low = q.lower()
            chosen_color = (chosen.get("color") or "").lower()
            chosen_desc = (chosen.get("description") or "").lower()
            ok = chosen_color in q_low or any(t in q_low for t in chosen_desc.split() if len(t) > 3)
            if not ok:
                print(f"  ⚠️  WARNING: chosen color/desc does NOT appear in question text!")
        print()

    print("=" * 80)
    print(f"Summary: resolved={n_resolved}  ambiguous={n_ambiguous}")
    print(f"  rule 1 (color)      : {n_rule_color}")
    print(f"  rule 2 (description): {n_rule_desc}")


if __name__ == "__main__":
    main()
