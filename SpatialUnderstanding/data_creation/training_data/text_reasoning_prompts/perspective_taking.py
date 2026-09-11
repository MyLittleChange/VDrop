"""Perspective-taking questions: from one of the two images' POV, where is
<object>?

Sample id prefix in the SFT jsonl is ``perspective_taking``.

POV mapping (verified empirically against the V5 normalized data: the row's
``angle`` field is the angle measured from the requested image's POV):

* ``asking_to=agent_1`` ⇒ requested POV is image 2 (camera_1_0).
* ``asking_to=agent_2`` ⇒ requested POV is image 1 (camera_0_0).

Both images go to the same model, so model-facing prompts refer only to
"image 1" / "image 2" and the requested image viewpoint.

Branch on whether the target is visible in the POV camera:

* CASE A (target visible in POV image): cite where it sits in that frame.
* CASE B (target NOT visible in POV image): triangulate using the other
  image and a shared landmark visible in BOTH images, with the relative
  camera pose as fallback.

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

    Perspective-taking uses the requested image viewpoint.
    * asking_to=agent_1 ⇒ POV image 2 / camera_1_0.
    * asking_to=agent_2 ⇒ POV image 1 / camera_0_0.
    """
    asking = meta.get("asking_to")
    if asking is None:
        # Older data without asking_to: fall back on whichever question is set.
        # user_1 asking ⇒ POV is image 2.
        if meta.get("user_1_question"):
            asking = "agent_1"
        elif meta.get("user_2_question"):
            asking = "agent_2"
        else:
            asking = "agent_1"
    if asking == "agent_1":
        return 2, "camera_1_0", "camera_0_0"
    return 1, "camera_0_0", "camera_1_0"


INSTRUCTIONS = (
    "Reasoning shape for PERSPECTIVE TAKING questions:\n"
    "1. Identify which image's POV the question asks about (image 1 or image 2). "
    "Use neutral wording like 'The question asks from image 2's viewpoint, so I "
    "use image 2 as the POV.' In the reasoning, use only neutral image-viewpoint "
    "wording and do not refer to the prompt source.\n"
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
    "3. Conclude with the matching direction word from the options. Make the "
    "POV step explicit so the reader sees which image you reasoned from. "
    "Direction words only — no degrees, no metres, no oracle field names."
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

    pov_hits = find_objects_by_category(visible, pov_cam, obj)
    other_hits = find_objects_by_category(visible, other_cam, obj)

    pov_pick = find_target_object(pov_hits, row_question)
    other_pick = find_target_object(other_hits, row_question)

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
        f"POV image (the one whose perspective the question asks about): image {pov_idx}   "
        f"(other image: image {other_idx})\n"
        f"Closest 8-way bucket from image {pov_idx}'s POV: {bucket}  "
        f"(oracle — use this to land on the right answer; do NOT cite the raw angle "
        f"{angle_str} in your trace)\n"
        f"Distance from image {pov_idx}'s camera to target: {dist_str} (oracle — do NOT cite)"
    )

    def _placement_block(pick: dict, image_idx: int) -> str:
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
        return (
            f"Target candidates in image {image_idx} (use the question text "
            f"to identify the right one):\n"
            f"{render_candidate_placements(pick['candidates'], image_idx)}"
        )

    if pov_has_target:
        # CASE A.
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

    # CASE B.
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
            "Target object: Floor Lamp\n"
            "POV image (the one whose perspective the question asks about): image 1   "
            "(other image: image 2)\n"
            "Closest 8-way bucket from image 1's POV: front  "
            "(oracle — use this to land on the right answer; do NOT cite the raw angle "
            "4.59° in your trace)\n"
            "Distance from image 1's camera to target: 4.27 m (oracle — do NOT cite)\n"
            "Target placement in POV image (image 1): mid centre\n"
            "Target placement in image 2: mid centre-right\n"
            "Visibility note: the target IS visible in the POV image — describe "
            "where it sits there and translate to a direction word."
        ),
        "question": "From the viewpoint of the first image, in which direction is the grey floor lamp located?",
        "options": ["left", "front", "right", "behind"],
        "correct_letter": "B",
        "reasoning": (
            "The question asks about the viewpoint of the first image, so I reason from "
            "image 1's POV. The grey floor lamp is visible in image 1: it sits roughly "
            "in the middle of the frame, directly along the camera's line of sight rather "
            "than off to either side. Image 2 confirms it is the same lamp, just seen "
            "from a different vantage. Centre-of-frame in the POV image translates to "
            "front. Among the four cardinal options, front is option B."
        ),
    },
    {
        # CASE B: target NOT visible in POV; uses shared landmark + cameras-facing-each-other pose.
        "scene_context": (
            "Target object: Bookshelf\n"
            "POV image (the one whose perspective the question asks about): image 2   "
            "(other image: image 1)\n"
            "Closest 8-way bucket from image 2's POV: behind  "
            "(oracle — use this to land on the right answer; do NOT cite the raw angle "
            "182.40° in your trace)\n"
            "Distance from image 2's camera to target: 5.10 m (oracle — do NOT cite)\n"
            "Target is NOT visible in image 2. It IS visible in image 1, placed at: "
            "mid centre.\n"
            "Shared landmark objects visible in BOTH images:\n"
            "  - Beige Sofa 1 (beige): image 2=mid centre, image 1=mid centre\n"
            "Relative camera pose: image 1's camera sits roughly behind of image 2's "
            "camera, looking back toward the POV camera."
        ),
        "question": "From the viewpoint of the second image, in which direction is the wooden bookshelf?",
        "options": ["front", "right", "behind", "left"],
        "correct_letter": "C",
        "reasoning": (
            "The question asks from image 2's viewpoint, so I use image 2 as the POV. "
            "The bookshelf does not appear in image 2 at all — it must lie outside "
            "that camera's field of view. Image 1 does show it, sitting in the centre "
            "of that frame. To work out how the two views relate I look at the beige "
            "sofa, which appears in BOTH images: it sits centre-frame in each one. "
            "That is the geometry of two cameras pointing at each other with a shared "
            "object between them — image 1's camera is positioned behind image 2's "
            "camera, aimed back. So whatever sits centre-frame in image 1 (the "
            "bookshelf) is directly behind image 2's camera. From image 2's POV the "
            "bookshelf is behind, which is option C."
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
