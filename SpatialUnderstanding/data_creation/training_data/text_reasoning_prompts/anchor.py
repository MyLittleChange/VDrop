"""Anchor questions: which option is visible in BOTH views.

Per-option visibility cannot be recovered reliably from the dataset row, because
the option strings have been paraphrased away from their original obj_id keys in
``visible_objects_with_descriptions.json``. We only trust the construction
guarantee from ``generate_anchor_questions`` in
MultiAgent_Spatial/question_generation_v2/generate_questions.py:

* The correct option is, by construction, the one visible in BOTH images.
* The 3 distractors are each visible in only ONE of the two images.

The model is fed both images and decides per-option visibility from them. We do
not surface "asking" vs "partner" framing — both images go to the same model,
the question is symmetric across views, and the per-side count would only be
useful if we also knew which option sits on which side (we don't, post-paraphrase).
"""

from .common import render_example, render_options

INSTRUCTIONS = (
    "Reasoning shape for ANCHOR questions:\n"
    "1. Recall that the correct anchor is the one option visible in BOTH images; "
    "every distractor is visible in only one of the two views.\n"
    "2. Scan each option in image 1 and image 2, calling out what you see.\n"
    "3. Eliminate the options visible in only one view, then conclude with the "
    "option that appears in both."
)


def render_scene_context(meta: dict, scene: dict) -> str:
    options = meta.get("options_user_1") or meta.get("options_user_2") or []
    cats = meta.get("option_categories") or [""] * len(options)

    rows = ["Letter | option | category"]
    rows.append("-------|--------|----------")
    for i, opt in enumerate(options):
        c = cats[i] if i < len(cats) else ""
        rows.append(f"{chr(65+i)}) | {opt} | {c}")

    return (
        "Option list:\n" + "\n".join(rows) + "\n\n"
        "Visibility ground truth (from the 3D scene the question was built from):\n"
        "- exactly ONE option is visible in BOTH images — that is the correct anchor;\n"
        "- each of the other options is visible in only ONE of the two images.\n"
        "Use the two images to identify which option is the shared one."
    )


EXAMPLES = [
    {
        "scene_context": (
            "Option list:\n"
            "Letter | option | category\n"
            "-------|--------|----------\n"
            "A) | yellow sofa | Sofa\n"
            "B) | black plant container next to a white shelf | Plant Container\n"
            "C) | black door | Door\n"
            "D) | window with white curtains next to a black shelf | Window\n\n"
            "Visibility ground truth (from the 3D scene the question was built from):\n"
            "- exactly ONE option is visible in BOTH images — that is the correct anchor;\n"
            "- each of the other options is visible in only ONE of the two images.\n"
            "Use the two images to identify which option is the shared one."
        ),
        "question": "Which of the following objects is present in both perspectives of the room?",
        "options": [
            "yellow sofa",
            "black plant container next to a white shelf",
            "black door",
            "window with white curtains next to a black shelf",
        ],
        "correct_letter": "B",
        "reasoning": (
            "I need the one option that appears in both images. In image 1 I can see a "
            "yellow sofa, a black plant container by a white shelf, and a window with "
            "white curtains. In image 2 I see the same black plant container by a white "
            "shelf, plus a black door. The yellow sofa and the window only show up in "
            "image 1; the black door only in image 2. The black plant container is the "
            "single object present in both images, so the shared anchor is option B."
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
