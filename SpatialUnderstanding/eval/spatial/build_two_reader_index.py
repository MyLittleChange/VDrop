#!/usr/bin/env python3
"""
Build the per-sample provenance JSON for the Two-Reader Informativeness
experiment: one row per (sample_id) with paths/text for each oracle view T.

Reads the existing oracle artifacts (T_td PNGs, T_pano PNGs, T_cor PNG pairs,
T_noise assignments JSON, T_cot JSONLs) and writes a single index that the
eval harness consumes.

Missing artifacts -> null. T_cot rows are only included when status=='ok'.
"""
import argparse
import json
from pathlib import Path

TEST_JSONS = {
    "anchor": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_anchor_normalized.json",
    "counting": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_counting_normalized.json",
    "relative_distance": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
    "relative_direction": "/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_direction_normalized.json",
}

EXPERIMENT_ROOT = Path("/path/to/scratch/VisualCoT/infinigen/two_reader_informativeness")
TCOT_ROOT = EXPERIMENT_ROOT / "text_thinking_approved_mcqs"

# T_cor v2: per-scene point-matching dots from annotate_shared_objects.py
COR_V2_ROOT = Path("/path/to/scratch/VisualCoT/infinigen/correspondence_shared_objects_approved_mcqs")
# T_td_blender: per-scene photoreal Blender ortho topdowns
TD_BLENDER_ROOT = Path("/path/to/scratch/VisualCoT/infinigen/topdown_blender_approved_mcqs")
# T_*_gemini: per-sample Gemini 3 Pro Image generations (pilot, 100 samples)
GEMINI_ROOT = EXPERIMENT_ROOT / "gemini_oracles"


def cor_paths_for(scene_id):
    """Per-scene T_cor (v2): annotated_cam{0,1}.png from annotate_shared_objects."""
    base = COR_V2_ROOT / scene_id
    v1 = base / "annotated_cam0.png"
    v2 = base / "annotated_cam1.png"
    if v1.exists() and v2.exists():
        return [str(v1), str(v2)]
    return None


def td_blender_path_for(scene_id):
    """Per-scene T_td_blender: photoreal Cycles ortho topdown."""
    p = TD_BLENDER_ROOT / scene_id / "topdown_blender.png"
    return str(p) if p.exists() else None


def td_gemini_path_for(scene_id, sample_id):
    p = GEMINI_ROOT / "topdown" / scene_id / f"topdown_gemini_{sample_id}.png"
    return str(p) if p.exists() else None


def pano_gemini_path_for(scene_id, sample_id):
    p = GEMINI_ROOT / "panorama" / scene_id / f"panorama_gemini_{sample_id}.png"
    return str(p) if p.exists() else None


def cor_gemini_paths_for(scene_id, sample_id):
    base = GEMINI_ROOT / "cor" / scene_id
    v1 = base / f"cor_gemini_{sample_id}_v1.png"
    v2 = base / f"cor_gemini_{sample_id}_v2.png"
    if v1.exists() and v2.exists():
        return [str(v1), str(v2)]
    return None


# T_td synth fallback dir (matplotlib BEV produced by generate_topdown_from_dataset.py)
TD_SYNTH_ROOT = Path("/path/to/scratch/VisualCoT/infinigen/topdown_maps_rel")


PANO_ROOT = Path("/path/to/scratch/VisualCoT/infinigen")


def pano_path_for(subtask, scene_id, sample_id):
    """Per-sample T_pano discovery: <root>/rendered_panorama_<subtask>_normalized/<scene>/panorama_blender_limits_<sample_id>.png."""
    p = PANO_ROOT / f"rendered_panorama_{subtask}_normalized" / scene_id / f"panorama_blender_limits_{sample_id}.png"
    return str(p) if p.exists() else None


def td_synth_path_for(subtask, scene_id, sample_id, asking_to):
    """Resolve T_td synth path via on-disk lookup when JSON's topdown_path is null."""
    base = TD_SYNTH_ROOT / f"topdown_approved_mcqs_{subtask}" / scene_id
    # asking_to may be 'agent_1' / 'agent_2' / None; prefer agent_2 (most common asker)
    agent_idx = 1 if asking_to == "agent_1" else 2
    candidate = base / f"topdown_agent_{agent_idx}_{sample_id}.png"
    if candidate.exists():
        return str(candidate)
    # Try the other agent if the preferred one isn't there
    other = base / f"topdown_agent_{3 - agent_idx}_{sample_id}.png"
    if other.exists():
        return str(other)
    return None


def load_tcot_for(subtask):
    """Load successfully-annotated text-CoT entries keyed by sample_id."""
    path = TCOT_ROOT / f"{subtask}.jsonl"
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for ln in f:
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok" and row.get("think_text"):
                out[row["sample_id"]] = row["think_text"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(EXPERIMENT_ROOT / "two_reader_informativeness_index.json"),
    )
    args = parser.parse_args()

    # Noise assignments
    noise_path = EXPERIMENT_ROOT / "noise_assignments.json"
    noise_map = {}
    if noise_path.exists():
        noise_map = json.load(open(noise_path))

    VIEW_KEYS = (
        "T_td", "T_td_blender", "T_pano", "T_cor", "T_cot", "T_noise",
        "T_td_gemini", "T_pano_gemini", "T_cor_gemini",
    )

    rows = []
    counters = {k: 0 for k in VIEW_KEYS}
    counters["total"] = 0
    per_subtask = {}

    for subtask, jp in TEST_JSONS.items():
        samples = json.load(open(jp))
        tcot = load_tcot_for(subtask)
        sub_count = {k: 0 for k in VIEW_KEYS}
        sub_count["total"] = len(samples)

        for s in samples:
            sid = s["sample_id"]
            scene = s["scene_id"]

            td = s.get("topdown_path")
            td = td if td and Path(td).exists() else None
            if td is None:
                # Fallback: discover on disk (handles JSON entries with null topdown_path)
                td = td_synth_path_for(subtask, scene, sid, s.get("asking_to"))

            td_blender = td_blender_path_for(scene)

            pano = s.get("panorama_path")
            pano = pano if pano and Path(pano).exists() else None
            if pano is None:
                pano = pano_path_for(subtask, scene, sid)

            cor = cor_paths_for(scene)
            cot = tcot.get(sid)
            noise = noise_map.get(sid)

            td_gemini = td_gemini_path_for(scene, sid)
            pano_gemini = pano_gemini_path_for(scene, sid)
            cor_gemini = cor_gemini_paths_for(scene, sid)

            row = {
                "sample_id": sid,
                "subtask": subtask,
                "scene_id": scene,
                "T_none": None,
                "T_cot": cot,
                "T_td": td,
                "T_td_blender": td_blender,
                "T_cor": cor,
                "T_pano": pano,
                "T_noise": noise,
                "T_td_gemini": td_gemini,
                "T_pano_gemini": pano_gemini,
                "T_cor_gemini": cor_gemini,
            }
            rows.append(row)

            for k in VIEW_KEYS:
                if row[k]:
                    counters[k] += 1
                    sub_count[k] += 1
            counters["total"] += 1

        per_subtask[subtask] = sub_count

    # Coverage summary
    summary = {
        "n_samples": counters["total"],
        "coverage_overall": {
            k: f"{counters[k]}/{counters['total']}" for k in VIEW_KEYS
        },
        "coverage_per_subtask": {
            subtask: {k: f"{v[k]}/{v['total']}" for k in VIEW_KEYS}
            for subtask, v in per_subtask.items()
        },
    }

    out_obj = {"summary": summary, "samples": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_obj, open(args.output, "w"), indent=2)
    print(f"wrote {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
