"""Spatial-orientation questions: from one of the two images' POV (the asker's
own camera), where is <object>?

Sample id prefix in the SFT jsonl is ``spatial`` (the dataset's ``question_type``
field is ``spatial_orientation``).

POV mapping (used by the V5 normalized rows):
* ``user_1_question`` is set ⇒ POV is image 1 (camera_0_0).
* ``user_2_question`` is set ⇒ POV is image 2 (camera_1_0).

Empirically ~56% of rows have the target object NOT visible in the POV camera
(the asker is asking about an off-screen object the partner / second image can
see). We therefore branch on visibility:

* CASE A (target visible in POV image): cite where it sits in that frame.
* CASE B (target NOT visible in POV image): triangulate using the OTHER image
  + a shared landmark visible in BOTH images (or, if no shared landmark
  exists, the relative camera pose) so the trace can argue which side of the
  POV camera the target lies on, including "behind" when it's off the back.

Raw angles / distances stay in the metadata as ORACLE info (marked "do NOT
cite") so the LLM can land on the right answer; the trace must reason from
visual cues only.
"""

from .common import (
    angle_to_bucket,
    bbox_to_placement,
    find_objects_by_category,
    find_target_object,
    relative_camera_pose,
    render_candidate_placements,
    render_example,
    render_options,
    shared_landmarks,
)


def _pov_cam_keys(meta: dict):
    """Return ``(pov_image_index, pov_cam_key, other_cam_key)``.

    Spatial: ``user_1_question`` set ⇒ POV image 1 / camera_0_0.
              ``user_2_question`` set ⇒ POV image 2 / camera_1_0.
    """
    if meta.get("user_1_question"):
        return 1, "camera_0_0", "camera_1_0"
    if meta.get("user_2_question"):
        return 2, "camera_1_0", "camera_0_0"
    # Fallback to the older convention (asking_to=agent_1 ⇒ POV image 1).
    asking = meta.get("asking_to") or "agent_1"
    if asking == "agent_1":
        return 1, "camera_0_0", "camera_1_0"
    return 2, "camera_1_0", "camera_0_0"


INSTRUCTIONS = (
    "Reasoning shape for SPATIAL ORIENTATION questions (own perspective):\n"
    "1. Identify the POV image from the metadata (image 1 or image 2). The "
    "OTHER image is a second vantage you can use as evidence.\n"
    "2. Check whether the target object is visible in the POV image.\n"
    "   - If YES: describe where it appears in the POV frame (e.g. 'centre' "
    "→ front, 'centre-right' → front-right, 'right edge' → right) and "
    "translate that placement directly to a direction word.\n"
    "   - If NO: the target is off-screen for the POV camera. The other "
    "image shows it (note where it sits there). Pick a SHARED landmark "
    "visible in BOTH images and note where it sits in each frame; that "
    "anchors how the two views relate. If no shared landmark exists, fall "
    "back to the relative-camera-pose hint that says where the OTHER "
    "camera sits behind/beside the POV camera. Use this to infer which "
    "side of the POV camera the target lies on — including whether it is "
    "behind the POV camera entirely.\n"
    "3. For 8-way questions use {front, front-right, right, behind-right, "
    "behind, behind-left, left, front-left}; for 4-way questions snap to "
    "{front, right, behind, left}.\n"
    "4. Conclude with the matching option using direction words only — no "
    "degrees, no metres, no oracle field names."
)


def render_scene_context(meta: dict, scene: dict, row_question: str = "") -> str:
    angle = meta.get("angle")
    distance = meta.get("distance")
    obj = meta.get("question_object", "")
    pov_idx, pov_cam, other_cam = _pov_cam_keys(meta)
    bucket = angle_to_bucket(angle) if angle is not None else "?"
    angle_str = f"{angle:.2f}°" if angle is not None else "?"
    dist_str = f"{distance:.2f} m" if distance is not None else "?"

    visible = scene.get("visible") or {}
    cameras = scene.get("cameras") or {}
    other_idx = 2 if pov_idx == 1 else 1

    # Locate target in each camera's visible list (case-insensitive name match).
    pov_hits = find_objects_by_category(visible, pov_cam, obj)
    other_hits = find_objects_by_category(visible, other_cam, obj)

    pov_pick = find_target_object(pov_hits, row_question)
    other_pick = find_target_object(other_hits, row_question)

    # "Target really in image X" means the picker either resolved a chosen
    # candidate, or stayed genuinely ambiguous between MULTIPLE candidates.
    # When the picker says "ambiguous because the only candidate's color
    # contradicts the question", the target the question asks about is NOT
    # visible in this camera, even though the bare category matches.
    def _target_present(pick):
        if not pick["candidates"]:
            return False
        if pick["ambiguous"] and len(pick["candidates"]) <= 1:
            return False
        return True

    pov_has_target = _target_present(pov_pick)
    other_has_target = _target_present(other_pick)

    header = (
        f"Target object: {obj}\n"
        f"POV image: image {pov_idx}   (other image: image {other_idx})\n"
        f"Closest 8-way bucket from image {pov_idx}'s POV: {bucket}  "
        f"(oracle — use this to land on the right answer; do NOT cite the raw angle "
        f"{angle_str} in your trace)\n"
        f"Distance from image {pov_idx}'s camera to target: {dist_str} (oracle — do NOT cite)"
    )

    def _placement_block(pick: dict, image_idx: int) -> str:
        """Format the target placement line(s) for one camera, listing all
        candidates when the picker can't disambiguate."""
        if not pick["candidates"]:
            return f"Target placement in image {image_idx}: not visible"
        if not pick["ambiguous"] and pick["chosen"] is not None:
            place = bbox_to_placement(pick["chosen"].get("bbox_2d"))["combined"]
            color = pick["chosen"].get("color") or ""
            label = f" ({color})" if color else ""
            return (
                f"Target placement in image {image_idx}: {place}"
                f"  [matched candidate: {pick['chosen'].get('name', '')}{label} — {pick['reason']}]"
            )
        # Ambiguous: render all candidates so the VLM can pick from the question.
        return (
            f"Target candidates in image {image_idx} (use the question text "
            f"to identify the right one):\n"
            f"{render_candidate_placements(pick['candidates'], image_idx)}"
        )

    if pov_has_target:
        # CASE A: target visible in POV.
        pov_block = _placement_block(pov_pick, pov_idx)
        other_block = _placement_block(other_pick, other_idx) if other_has_target else (
            f"Target placement in image {other_idx}: not visible"
        )
        return (
            f"{header}\n"
            f"{pov_block}\n"
            f"{other_block}\n"
            f"Visibility note: the target IS visible in the POV image — describe "
            f"where it sits there and translate to a direction word."
        )

    # CASE B: target NOT visible in POV image.
    if other_has_target:
        if not other_pick["ambiguous"] and other_pick["chosen"] is not None:
            place = bbox_to_placement(other_pick["chosen"].get("bbox_2d"))["combined"]
            color = other_pick["chosen"].get("color") or ""
            label = f" ({color})" if color else ""
            other_visibility = (
                f"Target is NOT visible in image {pov_idx}. It IS visible in image "
                f"{other_idx}, placed at: {place}  "
                f"[matched candidate: {other_pick['chosen'].get('name', '')}{label} — {other_pick['reason']}]."
            )
        else:
            other_visibility = (
                f"Target is NOT visible in image {pov_idx}. Multiple candidates "
                f"are visible in image {other_idx} — use the question text to "
                f"identify the right one:\n"
                f"{render_candidate_placements(other_pick['candidates'], other_idx)}"
            )
    else:
        other_visibility = (
            f"Target is NOT visible in image {pov_idx} and not located in image "
            f"{other_idx} either (it is in the room but off-screen for both cameras)."
        )

    landmarks = shared_landmarks(visible, pov_cam, other_cam)
    if landmarks:
        lm_lines = [
            f"  - {lm['name']}"
            + (f" ({lm['color']})" if lm.get("color") else "")
            + f": image {pov_idx}={lm['pov_placement']}, image {other_idx}={lm['other_placement']}"
            for lm in landmarks
        ]
        landmarks_block = "Shared landmark objects visible in BOTH images:\n" + "\n".join(lm_lines)
    else:
        landmarks_block = "No object is visible in both images."

    pose_block = ""
    T_pov = (cameras.get(pov_cam) or {}).get("T")
    T_oth = (cameras.get(other_cam) or {}).get("T")
    if T_pov is not None and T_oth is not None:
        pose = relative_camera_pose(T_pov, T_oth)
        pose_block = (
            f"\nRelative camera pose: image {other_idx}'s camera sits roughly "
            f"{pose['other_from_pov']} of image {pov_idx}'s camera, looking "
            f"{pose['facing']}."
        )

    return (
        f"{header}\n"
        f"{other_visibility}\n"
        f"{landmarks_block}"
        f"{pose_block}"
    )


EXAMPLES = [
    {
        # CASE A: target visible in POV.
        "scene_context": (
            "Target object: Window\n"
            "POV image: image 1   (other image: image 2)\n"
            "Closest 8-way bucket from image 1's POV: front-right  "
            "(oracle — use this to land on the right answer; do NOT cite the raw angle "
            "65.99° in your trace)\n"
            "Distance from image 1's camera to target: 3.15 m (oracle — do NOT cite)\n"
            "Target placement in POV image (image 1): mid centre-right\n"
            "Target placement in image 2: mid centre-left\n"
            "Visibility note: the target IS visible in the POV image — describe "
            "where it sits there and translate to a direction word."
        ),
        "question": "From your perspective, in which direction is the window with the beige curtains?",
        "options": ["behind-left", "front-right", "front-left", "behind-right"],
        "correct_letter": "B",
        "reasoning": (
            "The question is asked from image 1's POV, and the window with the beige "
            "curtains is visible in that frame: it sits in the right half of the image, "
            "still clearly forward (not behind), but offset to the right rather than "
            "dead ahead. That places it in the front-right region. Image 2 confirms it "
            "is the same window — same beige curtains, same wall — just seen from a "
            "different vantage. Among the options, front-right is option B."
        ),
    },
    {
        # CASE B: target NOT visible in POV; uses a shared landmark + relative camera pose.
        "scene_context": (
            "Target object: Refrigerator\n"
            "POV image: image 1   (other image: image 2)\n"
            "Closest 8-way bucket from image 1's POV: behind-left  "
            "(oracle — use this to land on the right answer; do NOT cite the raw angle "
            "215.00° in your trace)\n"
            "Distance from image 1's camera to target: 4.10 m (oracle — do NOT cite)\n"
            "Target is NOT visible in image 1. It IS visible in image 2, placed at: "
            "mid centre-left.\n"
            "Shared landmark objects visible in BOTH images:\n"
            "  - Wooden Dining Table 1 (brown): image 1=lower centre, image 2=lower right edge\n"
            "Relative camera pose: image 2's camera sits roughly behind of image 1's "
            "camera, looking back toward the POV camera."
        ),
        "question": "From your perspective, in which direction is the white refrigerator?",
        "options": ["front-right", "behind-left", "front-left", "behind"],
        "correct_letter": "B",
        "reasoning": (
            "The question is asked from image 1's POV. The refrigerator does not appear "
            "in image 1 at all, so it must lie outside that camera's field of view. "
            "Image 2 does show it — sitting in the centre-left of that frame. To work "
            "out where image 2 is shooting from, I look at the wooden dining table that "
            "appears in BOTH views: in image 1 it sits front-and-centre, in image 2 it "
            "is at the lower-right edge. That tells me image 2's camera is positioned "
            "behind image 1's camera and aimed back toward it. Given that pose, what is "
            "centre-left in image 2 lies behind and to the left of image 1's camera. "
            "So from image 1's POV the refrigerator is behind-left, which is option B."
        ),
    },
]


def build_prompt(meta: dict, scene: dict, row_question: str, options: list, gt_letter: str) -> str:
    scene_ctx = render_scene_context(meta, scene, row_question)
    examples = "\n\n".join(render_example(ex) for ex in EXAMPLES)
    opts = render_options(options)
    return (
        f"{INSTRUCTIONS}\n\n"
        f"--- IN-CONTEXT EXAMPLES ---\n{examples}\n--- END EXAMPLES ---\n\n"
        f"Now produce the reasoning for the following sample.\n\n"
        f"[scene metadata]\n{scene_ctx}\n\n"
        f"Question: {row_question}\n"
        f"Options:\n{opts}\n"
        f"Correct answer: {gt_letter}\n\n"
        f"Output exactly:\nReasoning:\n3-6 sentences of visual reasoning\nFinal: {gt_letter}"
    )
