"""Point-matching (same_object mode) reasoning prompts.

For every row in this dataset, REF in image 1 and the four candidates A/B/C/D
in image 2 sit on the SAME object — only the sub-anchor (centre, an outer
face, or a corner) differs. The reasoning shape is therefore:

  1) Identify the shared object (it is highlighted in both views).
  2) Locate the REF marker on the object's silhouette in image 1
     (top edge, lower-left corner, centre, ...).
  3) For each candidate in image 2, locate it on the same object's silhouette.
  4) Pick the candidate whose sub-position on the object matches REF's.

Oracle quantities surfaced to the annotator (anchor_type names, raw pixel
coordinates, object bboxes) are tagged "oracle — do NOT cite". The trace must
argue from VISUAL cues — qualitative grid placements within the object's
silhouette — that a student model could plausibly reproduce from the marked
images alone.
"""

from .common import render_example, render_options


INSTRUCTIONS = (
    "Reasoning shape for POINT-MATCHING (same-object) questions:\n"
    "1. Confirm that REF in image 1 and every candidate A/B/C/D in image 2 "
    "are stuck on the SAME object (the object name is given in the metadata).\n"
    "2. Describe where the REF marker sits on that object's silhouette in "
    "image 1 (e.g. 'near the upper-left corner of the lamp', 'on the "
    "right face', 'at the centre').\n"
    "3. Walk through each candidate marker in image 2 and describe where it "
    "sits on the SAME object's silhouette using the same kind of qualitative "
    "labels.\n"
    "4. Eliminate candidates whose sub-position on the object disagrees with "
    "the REF, and conclude with the candidate whose location matches."
)


# ---------------------------------------------------------------------------
# Per-marker placement labels (visual claim — citable in the trace)
# ---------------------------------------------------------------------------

_H_BUCKETS = ("left edge", "left side", "centre", "right side", "right edge")
_V_BUCKETS = ("top edge", "upper area", "middle", "lower area", "bottom edge")


def _placement_in_bbox(x: float, y: float, bbox_px) -> str:
    """Where (x, y) falls inside the pixel-space bbox [x1, y1, x2, y2].

    Returns a qualitative grid label like 'upper-left of the silhouette',
    'centre', 'on the right edge'. If the marker falls outside the bbox we
    fall back to 'just outside the silhouette' so the prompt never lies.
    """
    if not bbox_px or len(bbox_px) != 4:
        return "unknown placement on the object"
    x1, y1, x2, y2 = bbox_px
    w = max(1.0, float(x2) - float(x1))
    h = max(1.0, float(y2) - float(y1))
    fx = (float(x) - float(x1)) / w
    fy = (float(y) - float(y1)) / h
    if not (-0.05 <= fx <= 1.05 and -0.05 <= fy <= 1.05):
        return "just outside the object's silhouette"

    def bucket_h(f: float) -> str:
        if f < 0.15:
            return _H_BUCKETS[0]
        if f < 0.40:
            return _H_BUCKETS[1]
        if f <= 0.60:
            return _H_BUCKETS[2]
        if f <= 0.85:
            return _H_BUCKETS[3]
        return _H_BUCKETS[4]

    def bucket_v(f: float) -> str:
        if f < 0.15:
            return _V_BUCKETS[0]
        if f < 0.40:
            return _V_BUCKETS[1]
        if f <= 0.60:
            return _V_BUCKETS[2]
        if f <= 0.85:
            return _V_BUCKETS[3]
        return _V_BUCKETS[4]

    h_label = bucket_h(fx)
    v_label = bucket_v(fy)
    if h_label == "centre" and v_label == "middle":
        return "the centre of the silhouette"
    if h_label == "centre":
        return f"the {v_label} of the silhouette"
    if v_label == "middle":
        return f"the {h_label} of the silhouette"
    return f"the {v_label}, {h_label} of the silhouette"


def _bbox_norm_to_px(bbox_norm, hw):
    """Convert visible_objects.json bbox_2d (normalized, y bottom-origin) to
    pixel-space [x1, y1, x2, y2] with y top-origin (matches marker pixels).
    """
    if not bbox_norm or len(bbox_norm) != 4 or not hw or len(hw) != 2:
        return None
    h, w = float(hw[0]), float(hw[1])
    x_left = max(0.0, min(w - 1.0, float(bbox_norm[0]) * w))
    x_right = max(0.0, min(w - 1.0, float(bbox_norm[2]) * w))
    # bottom-origin → top-origin flip (matches strict_validator).
    y_top = max(0.0, min(h - 1.0, h - float(bbox_norm[3]) * h))
    y_bot = max(0.0, min(h - 1.0, h - float(bbox_norm[1]) * h))
    return [x_left, y_top, x_right, y_bot]


# ---------------------------------------------------------------------------
# Scene-context block
# ---------------------------------------------------------------------------

def render_scene_context(question_meta: dict, hw, visible_objects: dict) -> str:
    """Build the [scene metadata] block from the per-scene question entry.

    Inputs:
      question_meta : single entry of point_matching_questions[i]
      hw            : [H, W] from the QA json
      visible_objects : visible_objects.json dict {cam_key: {obj_key: {...}}}
    """
    proj = question_meta.get("projection_debug", {}) or {}
    cam_img1 = proj.get("cam_key_img1") or "camera_0_0"
    cam_img2 = proj.get("cam_key_img2") or "camera_1_0"
    ref_name = question_meta.get("ref_name") or "?"
    ref_object = question_meta.get("ref_object") or "?"

    # Object bbox in each view (oracle — used for placement labels).
    bbox_norm_img1 = (
        ((visible_objects.get(cam_img1) or {}).get(ref_name) or {}).get("bbox_2d")
    )
    bbox_norm_img2 = (
        ((visible_objects.get(cam_img2) or {}).get(ref_name) or {}).get("bbox_2d")
    )
    bbox_px_img1 = _bbox_norm_to_px(bbox_norm_img1, hw)
    bbox_px_img2 = _bbox_norm_to_px(bbox_norm_img2, hw)

    # REF marker placement (image 1).
    img1_markers = question_meta.get("img1_markers") or []
    ref_marker = next((m for m in img1_markers if m.get("label") == "REF"), None)
    if ref_marker is None and img1_markers:
        ref_marker = img1_markers[0]
    ref_x = float(ref_marker.get("x", 0.0)) if ref_marker else 0.0
    ref_y = float(ref_marker.get("y", 0.0)) if ref_marker else 0.0
    ref_placement_img1 = _placement_in_bbox(ref_x, ref_y, bbox_px_img1)

    # Candidate marker placements (image 2). Match by label A/B/C/D.
    img2_markers = {m.get("label"): m for m in (question_meta.get("img2_markers") or [])}
    cand_objs = question_meta.get("candidate_objects") or []
    rows_cands = []
    for c in cand_objs:
        label = c.get("label", "?")
        marker = img2_markers.get(label)
        if marker is None:
            placement = "unknown placement on the object"
        else:
            placement = _placement_in_bbox(
                float(marker.get("x", 0.0)), float(marker.get("y", 0.0)), bbox_px_img2
            )
        rows_cands.append(
            f"  - {label}) sits at {placement} of the same object in image 2 "
            f"(oracle — anchor_type {c.get('anchor_type', '?')}, do NOT cite "
            f"the anchor_type name)"
        )

    # Same-object guarantee + REF placement summary.
    out = [
        "Point-matching constraint (from the 3D scene the question was built from):",
        f"- The REF marker and every candidate A/B/C/D sit on the SAME object: "
        f"{ref_object} ({ref_name}). They differ only in WHERE on the object "
        f"the marker lands.",
        "",
        f"REF marker location in image 1: {ref_placement_img1} of the "
        f"{ref_object} (oracle — anchor_type "
        f"{(question_meta.get('ref_point_id') or '').split('::')[-1] or '?'}, "
        f"do NOT cite the anchor_type name).",
        "",
        "Candidate marker locations on the same object in image 2:",
        *rows_cands,
        "",
        "Use the two marked images to verify these placements (where each "
        "marker sits on the object's silhouette is a visual fact you can read "
        "off the image). Do NOT cite raw pixel coordinates, anchor_type names, "
        "or 'oracle' in your reasoning.",
    ]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# In-context example
# ---------------------------------------------------------------------------

EXAMPLES = [
    {
        "scene_context": (
            "Point-matching constraint (from the 3D scene the question was built from):\n"
            "- The REF marker and every candidate A/B/C/D sit on the SAME object: "
            "brown lamp (Floor Lamp 4). They differ only in WHERE on the object "
            "the marker lands.\n\n"
            "REF marker location in image 1: the bottom edge of the brown lamp "
            "(oracle — anchor_type face_ymin, do NOT cite the anchor_type name).\n\n"
            "Candidate marker locations on the same object in image 2:\n"
            "  - A) sits at the top edge of the same object in image 2 (oracle — "
            "anchor_type face_ymax, do NOT cite the anchor_type name)\n"
            "  - B) sits at the upper area, right side of the same object in image 2 "
            "(oracle — anchor_type corner45_5, do NOT cite the anchor_type name)\n"
            "  - C) sits at the lower area, right side of the same object in image 2 "
            "(oracle — anchor_type corner45_4, do NOT cite the anchor_type name)\n"
            "  - D) sits at the bottom edge of the same object in image 2 "
            "(oracle — anchor_type face_ymin, do NOT cite the anchor_type name)\n\n"
            "Use the two marked images to verify these placements ..."
        ),
        "question": (
            "A point is marked as REF in Image 1. Multiple candidate points "
            "(A, B, C, D) are marked in Image 2. Which point in Image 2 corresponds "
            "to the REF marker in Image 1?"
        ),
        "options": ["A", "B", "C", "D"],
        "correct_letter": "D",
        "reasoning": (
            "All five markers land on the same object — the brown floor lamp visible "
            "in both views. In image 1 the REF dot sits right at the bottom of the "
            "lamp's silhouette, against its base. I now check each candidate on the "
            "lamp in image 2: A is up at the top of the lamp, near its head, so it "
            "cannot match REF's base position. B is on the upper right of the lamp, "
            "and C is on the lower right corner — both off to the side, not "
            "centered at the bottom. D is the only marker at the very bottom edge "
            "of the lamp, in the same base region where REF sits in image 1. So D "
            "matches REF's sub-position on the shared object."
        ),
    },
]


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def build_prompt(
    question_meta: dict,
    hw,
    visible_objects: dict,
    row_question: str,
    options: list,
    gt_letter: str,
) -> str:
    scene_ctx = render_scene_context(question_meta, hw, visible_objects)
    examples = "\n\n".join(render_example(ex) for ex in EXAMPLES)
    opts = render_options(options)
    return (
        f"{INSTRUCTIONS}\n\n"
        f"--- IN-CONTEXT EXAMPLE ---\n{examples}\n--- END EXAMPLE ---\n\n"
        f"Now produce the reasoning for the following sample.\n\n"
        f"[scene metadata]\n{scene_ctx}\n\n"
        f"Question: {row_question}\n"
        f"Options:\n{opts}\n"
        f"Correct answer: {gt_letter}\n\n"
        f"Output exactly:\nReasoning:\n3-6 sentences of visual reasoning\nFinal: {gt_letter}"
    )
