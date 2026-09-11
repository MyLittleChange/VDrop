"""Relative-distance questions: which option is closest/farthest from the anchor."""

from .common import (
    bbox_to_placement,
    find_objects_by_category,
    relative_camera_pose,
    render_example,
    render_options,
    shared_landmarks,
)

INSTRUCTIONS = (
    "Reasoning shape for RELATIVE DISTANCE questions:\n"
    "1. State the anchor object (the reference) and whether the question asks for "
    "the closest or farthest option.\n"
    "2. Locate the anchor and each option in image 1 and image 2. Use visual "
    "placements like left/right/centre, foreground/background, same wall, same "
    "furniture area, and whether the object appears in one view or both.\n"
    "3. Before comparing options across different images, CONNECT THE TWO "
    "VIEWS: name a shared landmark visible in both images and explain how its "
    "placement changes between image 1 and image 2. If shared landmarks are "
    "missing or weak, use the relative-camera-pose hint to relate the two "
    "vantages.\n"
    "4. Compare each option's qualitative proximity to the anchor in the "
    "connected room layout. For closest, favor objects beside / on the same "
    "wall as the anchor; for farthest, favor objects across the room, deeper "
    "in the other view, or past nearer landmarks.\n"
    "5. Conclude with the matching option using qualitative depth language. Do "
    "NOT cite distances in metres, oracle field names, or that the answer was "
    "given — only terms like 'much closer', 'across the room', 'right next to', "
    "etc."
)


def _text_tokens(text: str) -> set:
    stop = {
        "a", "an", "and", "as", "at", "by", "in", "located", "near", "next",
        "of", "on", "the", "to", "with",
    }
    tokens = set()
    for tok in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split():
        if len(tok) <= 2 or tok in stop:
            continue
        if tok.endswith("s") and len(tok) > 4:
            tok = tok[:-1]
        tokens.add(tok)
    return tokens


def _color_tokens(tokens: set) -> set:
    colors = {
        "beige", "black", "blue", "brown", "gray", "green", "grey", "orange",
        "pink", "purple", "red", "white", "yellow",
    }
    return tokens & colors


def _object_label(obj: dict) -> str:
    name = obj.get("name") or "object"
    desc = obj.get("description") or ""
    color = obj.get("color") or ""
    details = ", ".join(x for x in (color, desc) if x)
    return f"{name} ({details})" if details else name


def _placement_for_obj(obj: dict) -> str:
    placement = bbox_to_placement(obj.get("bbox_2d"))["combined"]
    return f"{_object_label(obj)} at {placement}"


def _find_objects_by_text(visible: dict, cam_key: str, text: str) -> list:
    """Best-effort match against object name OR description."""
    hits = find_objects_by_category(visible, cam_key, text)
    seen = {oid for oid, _ in hits}
    text_low = (text or "").lower()
    for oid, obj in (visible.get(cam_key) or {}).items():
        if oid in seen:
            continue
        haystack = f"{obj.get('name') or ''} {obj.get('description') or ''}".lower()
        if text_low and text_low in haystack:
            hits.append((oid, obj))
    return hits


def _best_option_match(visible: dict, cam_key: str, category: str, option_text: str):
    """Return the best visible-object match for an option in one camera.

    Options are natural-language descriptions rather than stable object ids, so
    this intentionally stays best-effort and only reports matches with category
    support plus some overlap with the option wording.
    """
    option_tokens = _text_tokens(option_text)
    category_low = (category or "").lower()
    candidates = []
    for oid, obj in (visible.get(cam_key) or {}).items():
        haystack = f"{obj.get('name') or ''} {obj.get('description') or ''}".lower()
        category_ok = not category_low or category_low in haystack
        if not category_ok:
            continue
        haystack_tokens = _text_tokens(haystack)
        if _color_tokens(option_tokens) and not (_color_tokens(option_tokens) & haystack_tokens):
            continue
        overlap = len(option_tokens & haystack_tokens)
        if overlap == 0 and option_tokens:
            continue
        area = bbox_to_placement(obj.get("bbox_2d"))["area"]
        candidates.append((overlap, area, oid, obj))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], -item[1]))
    return candidates[0][3]


def _format_hits(visible: dict, cam_key: str, text: str, max_hits: int = 3) -> str:
    hits = _find_objects_by_text(visible, cam_key, text)
    if not hits:
        return "not located in visible-object metadata"
    return "; ".join(_placement_for_obj(obj) for _, obj in hits[:max_hits])


def _format_option_placement(visible: dict, cam_key: str, category: str, option_text: str) -> str:
    obj = _best_option_match(visible, cam_key, category, option_text)
    if obj is None:
        return "not located in visible-object metadata"
    return _placement_for_obj(obj)


def render_scene_context(meta: dict, scene: dict) -> str:
    anchor_obj = meta.get("question_object", "")
    qtype = meta.get("question_type", "")  # "closest" | "farthest"
    options = meta.get("options_user_1") or meta.get("options_user_2") or []
    distances = meta.get("option_distances") or []
    cats = meta.get("option_categories") or []
    in_view = meta.get("ans_present_in_view")
    distrib = meta.get("agent_distribution")
    visible = scene.get("visible") or {}
    cameras = scene.get("cameras") or {}

    rows = [
        "Letter | option | category | oracle distance to anchor | image 1 placement | image 2 placement"
    ]
    rows.append(
        "-------|--------|----------|---------------------------|-------------------|------------------"
    )
    for i, opt in enumerate(options):
        d = distances[i] if i < len(distances) else float("nan")
        c = cats[i] if i < len(cats) else ""
        img1_place = _format_option_placement(visible, "camera_0_0", c, opt)
        img2_place = _format_option_placement(visible, "camera_1_0", c, opt)
        rows.append(
            f"{chr(65+i)}) | {opt} | {c} | {d:.3f} m | {img1_place} | {img2_place}"
        )

    landmarks = shared_landmarks(visible, "camera_0_0", "camera_1_0")
    if landmarks:
        lm_lines = [
            f"  - {lm['name']}"
            + (f" ({lm['color']})" if lm.get("color") else "")
            + f": image 1={lm['pov_placement']}, image 2={lm['other_placement']}"
            for lm in landmarks
        ]
        landmarks_block = "Shared landmark objects visible in BOTH images:\n" + "\n".join(lm_lines)
    else:
        landmarks_block = "No strong shared landmark was located in both images."

    pose_block = ""
    T1 = (cameras.get("camera_0_0") or {}).get("T")
    T2 = (cameras.get("camera_1_0") or {}).get("T")
    if T1 is not None and T2 is not None:
        pose = relative_camera_pose(T1, T2)
        pose_block = (
            "\nRelative camera pose: image 2's camera sits roughly "
            f"{pose['other_from_pov']} of image 1's camera, looking "
            f"{pose['facing']}."
        )

    return (
        f"Anchor (reference) object: {anchor_obj}\n"
        f"Question type: {qtype} (we want the {'closest' if qtype == 'closest' else 'farthest'} option)\n"
        f"Answer is visible to the asking camera: {in_view}\n"
        f"Agent distribution of options across cameras: {distrib}\n"
        f"Anchor placement in image 1: {_format_hits(visible, 'camera_0_0', anchor_obj)}\n"
        f"Anchor placement in image 2: {_format_hits(visible, 'camera_1_0', anchor_obj)}\n"
        f"{landmarks_block}"
        f"{pose_block}\n\n"
        + "\n".join(rows) + "\n"
        "(The oracle distances above are a private cheat sheet to help you land on "
        "the right option — do NOT cite them in metres in your trace; reason about "
        "visual proximity instead. Use the placement and shared-landmark blocks to "
        "connect image 1 and image 2 before comparing options across views.)"
    )


EXAMPLES = [
    {
        "scene_context": (
            "Anchor (reference) object: Shelf\n"
            "Question type: farthest (we want the farthest option)\n"
            "Answer is visible to the asking camera: True\n"
            "Agent distribution of options across cameras: 3-1\n"
            "Anchor placement in image 1: T V Stand 1 (white, white Shelf) at mid centre-right\n"
            "Anchor placement in image 2: T V Stand 1 (white, white Shelf) at mid centre-left\n"
            "Shared landmark objects visible in BOTH images:\n"
            "  - T V Stand 1 (white): image 1=mid centre-right, image 2=mid centre-left\n"
            "  - Large Plant Container 1 (black): image 1=mid right edge, image 2=mid centre-left\n"
            "Relative camera pose: image 2's camera sits roughly behind-right of image 1's camera, "
            "looking back toward the image 1 camera.\n\n"
            "Letter | option | category | oracle distance to anchor | image 1 placement | image 2 placement\n"
            "-------|--------|----------|---------------------------|-------------------|------------------\n"
            "A) | window with white curtains located next to a black shelf | Window | 1.800 m | Window 9 (white curtain, white curtain Window next to a black Shelf) at mid centre | not located in visible-object metadata\n"
            "B) | shelf located near a green door | Shelf | 5.282 m | Cell Shelf 1 (black, Shelf near a green Door) at mid centre-left | not located in visible-object metadata\n"
            "C) | door located near a black shelf | Door | 5.984 m | Glass Panel Door 4 (green, Door near a black Shelf) at mid left edge | Glass Panel Door 2 (black, black Door) at mid centre\n"
            "D) | yellow sofa | Sofa | 3.149 m | not located in visible-object metadata | Sofa 1 (yellow, yellow Sofa) at upper right edge\n"
            "(The oracle distances above are a private cheat sheet to help you land on "
            "the right option — do NOT cite them in metres in your trace; reason about "
            "visual proximity instead. Use the placement and shared-landmark blocks to "
            "connect image 1 and image 2 before comparing options across views.)"
        ),
        "question": "Which of the following objects is situated furthest from a white shelf?",
        "options": [
            "window with white curtains located next to a black shelf",
            "shelf located near a green door",
            "door located near a black shelf",
            "yellow sofa",
        ],
        "correct_letter": "C",
        "reasoning": (
            "The reference is the white shelf, and I want the option that sits farthest "
            "from it. In image 1, the white shelf is visible on the right side, and the "
            "window in option A is on the same nearby wall area, so A is clearly close "
            "to the reference. To compare across the two views, I connect image 1 and "
            "image 2 using the shared white shelf and black plant container: the same "
            "objects shift from the right side of image 1 to the left/centre-left of "
            "image 2, showing that image 2 is looking back across the same room from a "
            "different side. With that connection, the yellow sofa in image 2 is away "
            "from the shelf area but still in the open middle of the room. The shelf "
            "near the green door and the door near a black shelf sit on the far side "
            "of the connected layout, and the door in option C is deeper/past the "
            "nearer shelf area. So the farthest option is C."
        ),
    },
]


def build_prompt(meta: dict, scene: dict, row_question: str, options: list, gt_letter: str) -> str:
    scene_ctx = render_scene_context(meta, scene)
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
