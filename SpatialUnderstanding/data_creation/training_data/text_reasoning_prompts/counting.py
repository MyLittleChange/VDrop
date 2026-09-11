"""Counting questions: how many <category> are visible across both views."""

from .common import SHARED_SYSTEM, render_example, render_options

INSTRUCTIONS = (
    "Reasoning shape for COUNTING questions:\n"
    "1. State the object category being counted.\n"
    "2. Enumerate the instances of that category visible in image 1, then image 2, "
    "naming each by a short visual descriptor (colour, neighbour object, etc.).\n"
    "3. Reconcile overlap VISUALLY — if the same instance is plainly the same "
    "object across both views (same colour / surroundings / pose), count it "
    "once, not twice.\n"
    "4. Note that some instances may exist in the room but lie outside both camera "
    "frames; the question asks about the entire room.\n"
    "5. Conclude with the room total and pick the matching option. Do NOT cite "
    "oracle field names like difficulty_int / difficulty_sum — reason about "
    "the images directly."
)


def render_scene_context(meta: dict, scene: dict) -> str:
    """meta = a row from dataset_counting_questions_filtered_V*_normalized.json
    scene = dict with `visible` (visible_objects_with_descriptions.json contents).
    """
    target_cat = meta.get("question_object", "")
    diff_sum = meta.get("difficulty_sum")
    diff_int = meta.get("difficulty_int")
    room_total = meta.get("correct_answer")  # e.g. "3"

    visible = scene.get("visible") or {}
    cam_lines = []
    for cam_key, objs in sorted(visible.items()):
        names = [
            o.get("description") or o.get("name", "")
            for o in objs.values()
            if (o.get("name", "").lower().startswith(target_cat.lower())
                or target_cat.lower() in o.get("name", "").lower())
        ]
        cam_lines.append(
            f"- {cam_key}: {len(names)} {target_cat.lower()}(s)"
            + (f" — {', '.join(names)}" if names else "")
        )

    return (
        f"Object category being counted: {target_cat}\n"
        f"Per-camera visibility (what each agent sees of this category):\n"
        + "\n".join(cam_lines) + "\n"
        f"Oracle counts (private cheat sheet — do NOT cite these field names in your trace):\n"
        f"- {diff_sum} total category sightings summed across the two cameras\n"
        f"- {diff_int} instance(s) of this category appear in BOTH images at once\n"
        f"- {room_total} is the ground-truth room total (the answer; some instances "
        "may lie outside both camera frames)."
    )


EXAMPLES = [
    {
        "scene_context": (
            "Object category being counted: Window\n"
            "Per-camera visibility (what each agent sees of this category):\n"
            "- camera_0_0: 2 window(s) — window with the brown curtains, window with the white curtains\n"
            "- camera_1_0: 2 window(s) — window with the white curtains, window above the green dishwasher\n"
            "Oracle counts (private cheat sheet — do NOT cite these field names in your trace):\n"
            "- 4 total category sightings summed across the two cameras\n"
            "- 1 instance(s) of this category appear in BOTH images at once\n"
            "- 3 is the ground-truth room total (the answer; some instances may lie "
            "outside both camera frames)."
        ),
        "question": "What is the total number of windows in the room?",
        "options": ["4", "3", "1", "2"],
        "correct_letter": "B",
        "reasoning": (
            "I am asked to count windows across the whole room. In image 1 I see two "
            "windows — one with brown curtains and one with white curtains. In image 2 "
            "I also see two windows — one with white curtains and one above the green "
            "dishwasher. The window with white curtains looks identical in both views "
            "(same curtains, same neighbouring wall) — it is one window seen from two "
            "angles, not two separate windows. So the unique windows I have evidence "
            "for are: brown curtains, white curtains, and above the dishwasher — three "
            "in total. The room total is 3, which is option B."
        ),
    },
]


def build_prompt(meta: dict, scene: dict, row_question: str, options: list, gt_letter: str) -> str:
    """Returns the user-side text content (the system prompt is sent separately)."""
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
