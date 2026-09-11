"""Shared prompt scaffolding for the text-reasoning annotator."""

import numpy as np

SHARED_SYSTEM = (
    "You annotate spatial-reasoning training data. You are shown two camera "
    "images, ground-truth scene metadata extracted from a 3D engine, a "
    "multiple-choice question, and the correct answer.\n\n"
    "The metadata's numeric quantities (raw angles in degrees, raw distances in "
    "metres, count fields like difficulty_int / difficulty_sum) are ORACLE "
    "information — given to you so you can land on the right answer with "
    "confidence. Treat them as a private cheat sheet.\n\n"
    "The Reasoning section you write must be reasoning a future model — looking "
    "only at the two images and the question — could plausibly produce. Argue from "
    "VISUAL cues: where the object appears in each frame (centre / left / right "
    "edge / not visible), what is foreground vs. background, what is the same "
    "object seen from two angles, which view contains the anchor. Use direction "
    "words ('front', 'front-right', etc.) and qualitative depth comparisons "
    "('appears closer than', 'sits behind'), NOT raw numeric quantities. Do not "
    "cite angles in degrees, distances in metres, or oracle field names. Do not "
    "state that you were told the answer.\n\n"
    "When the metadata lists MULTIPLE candidate instances for the target object "
    "(e.g. two doors, two sofas), use the question text — its color word, its "
    "neighbor-object phrase ('next to a green door'), or any location qualifier "
    "— to pick the correct instance. Do NOT just describe the first candidate "
    "listed. If the question gives no disambiguator, reason about which "
    "candidate the question most likely refers to and say so.\n\n"
    "Do not use XML tags or tool calls. Output plain text with a Reasoning "
    "section and a Final line."
)


def render_options(options):
    return "\n".join(f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options))


def render_example(example: dict) -> str:
    """Render a single in-context example as a complete user→assistant turn."""
    opts = render_options(example["options"])
    user = (
        f"[scene metadata]\n{example['scene_context']}\n\n"
        f"Question: {example['question']}\n"
        f"Options:\n{opts}\n"
        f"Correct answer: {example['correct_letter']}\n"
    )
    assistant = f"Reasoning:\n{example['reasoning']}\nFinal: {example['correct_letter']}"
    return f"USER:\n{user}\nASSISTANT:\n{assistant}\n"


def angle_to_bucket(angle_deg: float) -> str:
    """Convert a 0..360 angle (0 = front, 90 = right, 180 = behind, 270 = left)
    into one of the 8 orientation buckets used by the dataset.

    The dataset uses 8 directions: front, behind, left, right, front-left,
    front-right, behind-left, behind-right. 4-way questions only use the cardinal
    four. We return the closest bucket name.
    """
    a = angle_deg % 360
    if a <= 22.5 or a >= 337.5:
        return "front"
    if a < 67.5:
        return "front-right"
    if a <= 112.5:
        return "right"
    if a < 157.5:
        return "behind-right"
    if a <= 202.5:
        return "behind"
    if a < 247.5:
        return "behind-left"
    if a <= 292.5:
        return "left"
    return "front-left"


# --- Visual placement / camera-geometry helpers ----------------------------
# These let us cite where an object APPEARS in each image (a visual claim a
# student model can verify) instead of citing raw oracle angles or distances.


def bbox_to_placement(bbox_2d) -> dict:
    """Map a normalized [x1, y1, x2, y2] bbox (x increases rightward, y
    increases downward — top-left is 0,0) to qualitative placement labels.

    Returns ``{"h": ..., "v": ..., "combined": ..., "area": ...}``. ``area``
    is the bbox area as a fraction of the image, useful for sorting shared
    landmarks by visual prominence.
    """
    if not bbox_2d or len(bbox_2d) != 4:
        return {"h": "?", "v": "?", "combined": "?", "area": 0.0}
    x1, y1, x2, y2 = bbox_2d
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    if cx < 0.15:
        h = "left edge"
    elif cx < 0.40:
        h = "centre-left"
    elif cx < 0.60:
        h = "centre"
    elif cx < 0.85:
        h = "centre-right"
    else:
        h = "right edge"
    if cy < 0.33:
        v = "upper"
    elif cy < 0.66:
        v = "mid"
    else:
        v = "lower"
    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    combined = f"{v} {h}" if h != "?" else "?"
    return {"h": h, "v": v, "combined": combined, "area": float(area)}


def angle_in_cam_frame(T_world_to_cam, world_xyz) -> float:
    """Angle from a camera to a world point, in the dataset convention.

    ``T_world_to_cam`` is the 4×4 stored in cameras.json (verified
    convention: T transforms world points into camera frame). The dataset's
    ``angle`` is computed as ``(degrees(arctan2(p_cam[0], -p_cam[2])) + 360)
    % 360`` with 0=front, 90=right, 180=behind, 270=left.
    """
    p = np.append(np.asarray(world_xyz, dtype=float), 1.0)
    p_cam = np.asarray(T_world_to_cam, dtype=float) @ p
    x, z = float(p_cam[0]), float(p_cam[2])
    return float((np.degrees(np.arctan2(x, -z)) + 360.0) % 360.0)


def relative_camera_pose(T_pov, T_other) -> dict:
    """Where the OTHER camera sits relative to the POV camera, plus which
    direction it is facing.

    Both ``T_pov`` and ``T_other`` are world-to-camera 4×4 matrices. We
    recover each camera's world position via ``-R^T @ t`` and its forward
    direction via the third row of R (since the camera looks down -Z and
    points at world direction ``-R^T @ e_z``; equivalently, the world
    direction of cam-forward equals the negated third column of R^T which
    is ``-R[2, :]``... but we just compute it numerically).

    Returns ``{"other_from_pov": <bucket>, "facing": <coarse facing label>}``.
    """
    T_pov = np.asarray(T_pov, dtype=float)
    T_other = np.asarray(T_other, dtype=float)
    R_pov, t_pov = T_pov[:3, :3], T_pov[:3, 3]
    R_oth, t_oth = T_other[:3, :3], T_other[:3, 3]
    pos_pov = -R_pov.T @ t_pov
    pos_oth = -R_oth.T @ t_oth

    # Bucket of the other camera's WORLD position from the POV camera's frame.
    other_from_pov_bucket = angle_to_bucket(angle_in_cam_frame(T_pov, pos_oth))

    # Forward direction in world for the other camera. World-to-cam means
    # cam-frame -Z (forward) corresponds to world direction -R_other^T @ e_z,
    # i.e. -R_other[2, :] expressed as a world vector via R_other^T @ (-e_z).
    fwd_oth_world = -R_oth.T @ np.array([0.0, 0.0, 1.0])
    fwd_oth_world = fwd_oth_world / (np.linalg.norm(fwd_oth_world) + 1e-9)

    # POV camera's own forward direction in world.
    fwd_pov_world = -R_pov.T @ np.array([0.0, 0.0, 1.0])
    fwd_pov_world = fwd_pov_world / (np.linalg.norm(fwd_pov_world) + 1e-9)

    dot = float(np.dot(fwd_oth_world, fwd_pov_world))
    if dot > 0.5:
        facing = "in roughly the same direction as the POV camera"
    elif dot < -0.5:
        facing = "back toward the POV camera"
    else:
        facing = "sideways relative to the POV camera"

    return {
        "other_from_pov": other_from_pov_bucket,
        "facing": facing,
    }


def find_objects_by_category(visible: dict, cam_key: str, target_cat: str) -> list:
    """Return list of (obj_key, obj_dict) for visible objects in ``cam_key``
    whose ``name`` matches ``target_cat`` (case-insensitive substring)."""
    if not target_cat:
        return []
    target_low = target_cat.lower()
    out = []
    for oid, obj in (visible.get(cam_key) or {}).items():
        name = (obj.get("name") or "").lower()
        if target_low in name or name.startswith(target_low):
            out.append((oid, obj))
    return out


_COLOR_WORDS = frozenset({
    "beige", "black", "blue", "brown", "gray", "green", "grey", "orange",
    "pink", "purple", "red", "tan", "white", "yellow", "wooden", "metal",
    "metallic", "silver", "gold", "golden",
})

_QUESTION_STOPWORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "in", "is", "it", "located", "near",
    "next", "of", "on", "or", "the", "to", "with", "which", "what", "where",
    "from", "your", "perspective", "viewpoint", "image", "first", "second",
    "direction", "object", "objects", "respect", "relative", "based", "view",
    "seen", "you", "are", "be", "this", "that", "between",
})


def _tokens(text: str) -> set:
    """Lowercase alnum tokens (length > 2) with simple plural-strip, minus stopwords."""
    if not text:
        return set()
    raw = "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
    out = set()
    for tok in raw:
        if len(tok) <= 2 or tok in _QUESTION_STOPWORDS:
            continue
        if tok.endswith("s") and len(tok) > 4:
            tok = tok[:-1]
        out.add(tok)
    return out


def find_target_object(candidates: list, question_text: str) -> dict:
    """Disambiguate among ``candidates`` (list of (oid, obj) from
    :func:`find_objects_by_category`) using the question text.

    Returns a dict ``{"chosen": <obj_or_None>, "candidates": <list of obj>,
    "ambiguous": bool, "reason": <str>}`` so callers can either render a single
    target placement or list all candidates when picking is ambiguous.

    Selection rules in priority order:
      1. Exact color match — if the question mentions a color word and exactly
         one candidate's ``color`` field matches that word, pick it.
      2. Description-token overlap — score each candidate by overlap of its
         ``description`` tokens with the question tokens; pick the unique top
         scorer when score >= 2 and beats #2 by >= 1.
      3. Otherwise mark ambiguous and return all candidates.

    With 0 or 1 candidates the function is trivial:
    - 0 candidates → ``{"chosen": None, "candidates": [], "ambiguous": False}``
    - 1 candidate  → ``{"chosen": that, "candidates": [that], "ambiguous": False}``
    """
    objs = [obj for _, obj in candidates]
    if not objs:
        return {"chosen": None, "candidates": [], "ambiguous": False, "reason": "no candidates"}

    q_tokens = _tokens(question_text)
    q_colors = q_tokens & _COLOR_WORDS

    if len(objs) == 1:
        # Single candidate: still verify its color doesn't contradict the
        # question. If the question says "pink door" and the only visible
        # door is orange, this isn't actually the door the question refers
        # to — flag as ambiguous so the caller doesn't pin its bbox-derived
        # placement to the wrong instance.
        only = objs[0]
        if q_colors:
            only_color = _tokens(only.get("color") or "") & _COLOR_WORDS
            if only_color and not (only_color & q_colors):
                return {
                    "chosen": None,
                    "candidates": objs,
                    "ambiguous": True,
                    "reason": "single candidate color contradicts question",
                }
        return {"chosen": only, "candidates": objs, "ambiguous": False, "reason": "single candidate"}

    def _score_desc(pool):
        """Return [(overlap, obj), ...] sorted descending."""
        scored = []
        for obj in pool:
            desc_tokens = _tokens(obj.get("description") or "")
            name_tokens = _tokens(obj.get("name") or "")
            scored.append((len((desc_tokens | name_tokens) & q_tokens), obj))
        scored.sort(key=lambda t: -t[0])
        return scored

    # Rule 0: STRONG description overlap (score >= 3 and beats #2 by >= 2).
    # This handles questions where a color word refers to a NEIGHBOR, not the
    # target — e.g. "the bed near a blue shelf", "the window next to a white
    # desk" — by giving description matches priority over the surface-level
    # color filter.
    strong = _score_desc(objs)
    if strong and strong[0][0] >= 3 and (len(strong) == 1 or strong[0][0] - strong[1][0] >= 2):
        return {
            "chosen": strong[0][1],
            "candidates": objs,
            "ambiguous": False,
            "reason": f"description overlap (score {strong[0][0]})",
        }

    # Rule 1: unique color match.
    if q_colors:
        color_hits = []
        for obj in objs:
            obj_color_toks = _tokens(obj.get("color") or "")
            if obj_color_toks & q_colors:
                color_hits.append(obj)
        if len(color_hits) == 1:
            return {
                "chosen": color_hits[0],
                "candidates": objs,
                "ambiguous": False,
                "reason": f"unique color match ({sorted(q_colors)[0]})",
            }
        if len(color_hits) > 1:
            objs_for_desc = color_hits
        else:
            objs_for_desc = objs
    else:
        objs_for_desc = objs

    # Rule 1b: when q has a color and rule 1 didn't find a unique hit, drop
    # candidates whose OWN color word contradicts the question's color so
    # rule 2 can't pick a wrong-color instance via category-name tokens.
    # If the filter wipes out everything, mark ambiguous.
    if q_colors:
        filtered = []
        for obj in objs_for_desc:
            obj_color_words = _tokens(obj.get("color") or "") & _COLOR_WORDS
            if obj_color_words and not (obj_color_words & q_colors):
                continue
            filtered.append(obj)
        if not filtered:
            return {
                "chosen": None,
                "candidates": objs,
                "ambiguous": True,
                "reason": f"{len(objs)} candidates, question color word matched no candidate",
            }
        objs_for_desc = filtered

    # Rule 2: description-token overlap (relaxed threshold).
    scored = _score_desc(objs_for_desc)
    if scored and scored[0][0] >= 2 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 1):
        return {
            "chosen": scored[0][1],
            "candidates": objs,
            "ambiguous": False,
            "reason": f"description overlap (score {scored[0][0]})",
        }

    return {
        "chosen": None,
        "candidates": objs_for_desc if len(objs_for_desc) < len(objs) else objs,
        "ambiguous": True,
        "reason": f"{len(objs)} candidates, no decisive signal",
    }


def render_candidate_placements(candidates: list, image_idx: int) -> str:
    """Render multiple candidates as a list of `name (color, desc) at placement` lines."""
    lines = []
    for obj in candidates:
        name = obj.get("name") or "object"
        color = obj.get("color") or ""
        desc = obj.get("description") or ""
        details = ", ".join(x for x in (color, desc) if x)
        label = f"{name} ({details})" if details else name
        place = bbox_to_placement(obj.get("bbox_2d"))["combined"]
        lines.append(f"  - {label} at {place}")
    return "\n".join(lines) if lines else f"  (no candidates in image {image_idx})"


def shared_landmarks(visible: dict, pov_cam: str, other_cam: str, top_k: int = 2,
                     min_area_in_pov: float = 0.01) -> list:
    """Return up to ``top_k`` objects visible in BOTH cameras, ranked by
    POV-image bbox area (largest first), filtered to those whose POV bbox
    area is at least ``min_area_in_pov`` of the frame.

    Each returned entry is a dict with ``name``, ``color``, ``description``,
    ``pov_placement`` (combined string), ``other_placement`` (combined string),
    and ``pov_area``.
    """
    pov_objs = visible.get(pov_cam) or {}
    other_objs = visible.get(other_cam) or {}
    candidates = []
    for oid, pov_o in pov_objs.items():
        if oid not in other_objs:
            continue
        pov_p = bbox_to_placement(pov_o.get("bbox_2d"))
        if pov_p["area"] < min_area_in_pov:
            continue
        oth_p = bbox_to_placement(other_objs[oid].get("bbox_2d"))
        candidates.append({
            "name": pov_o.get("name") or "",
            "color": pov_o.get("color"),
            "description": pov_o.get("description"),
            "pov_placement": pov_p["combined"],
            "other_placement": oth_p["combined"],
            "pov_area": pov_p["area"],
        })
    candidates.sort(key=lambda d: -d["pov_area"])
    return candidates[:top_k]
