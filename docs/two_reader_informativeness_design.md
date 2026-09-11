# Two-Reader Informativeness Experiment — Design Spec

**Goal.** Measure $I(T)$, the informativeness of each visual-thought type
$T$, in a way that does not depend on which model we chose as the reader.
Output: a 2D scatter where the x-axis is informativeness measured by a
vanilla in-family reader (BAGEL) and the y-axis is informativeness
measured by an architecturally independent strong reader (e.g.
Qwen3-VL-72B). If the points line up roughly diagonally, "$T$ is
informative" is a property of the representation; if they do not, it is
a reader artifact.

This document is the implementation contract: data, conditions, prompts,
metrics, sanity checks, reporting.

---

## 1. View types under test

| ID | Name | Render source | Notes |
|----|------|---------------|-------|
| $T_0$ | None (baseline) | n/a | No extra view fed. Defines $\mathrm{Acc}_{R,\mathrm{base}}$. |
| $T_{\mathrm{cot}}$ | Text CoT | LLM-written rationale (Qwen3-VL-235B-FP8 conditioned on $V_1, V_2$, question, gold) | A textual intermediate; same role as a "view" in the framework. Append before the answer. Quality filter: drop samples where the LLM's predicted answer ≠ gold. |
| $T_{\mathrm{td}}$ | Top-down | Infinigen BEV renderer | Bird's-eye of the same scene, scaled to the input camera coverage. |
| $T_{\mathrm{cor}}$ | Correspondence | 2D bbox overlay on $V_1, V_2$ from `visible_objects.json` | Up to 3 co-visible objects per question, colored consistently across both views. Built from existing per-scene bbox data, no new render. |
| $T_{\mathrm{pano}}$ | Panorama | Infinigen 360° render at $V_1$ position | Stitched panoramic view spanning $V_1$ and $V_2$ coverage. Coverage ≈ 85% of test split (146 / 1000 missing — see §3). |
| $T_{\mathrm{noise}}$ | Random unrelated image | Infinigen render of a different scene | **Sanity / null condition.** Should give $I_R \approx 0$. |

All renders come from Infinigen ground truth. We never use model-generated
views in this experiment — that lives in the *Learnability* experiment.

## 2. Readers

| Reader | Role | Why |
|--------|------|-----|
| Vanilla BAGEL | In-family reader | Same architecture you fine-tune on; tells you "how much spatial content does $T$ expose to a model in your family before any spatial training." |
| Qwen3-VL-235B-A22B-Instruct-FP8 | Independent strong reader | Architecturally unrelated, already deployed locally as the MMSI judge (4-GPU vLLM); tells you whether $T$'s informativeness is intrinsic or BAGEL-specific. |

Both readers must be **untouched on COSMIC / spatial reasoning**. Use the
released base BAGEL weights (no spatial fine-tuning), and an off-the-shelf
external VLM API.

Log the served model name (`qwen3_vl_235b_fp8`) and the local checkpoint
path so the experiment is reproducible.

## 3. Evaluation set

- **Primary**: COSMIC test split (`approved_mcqs_*_normalized.json`),
  4 subtasks × 250 samples = 1000 items. Rotation is **not** part of
  COSMIC and is excluded from this experiment (it lives in the
  Matterport pipeline; see §6).
- **Transport (small, optional)**: ~100 hand-picked examples from
  MMSI-Bench / MindCube where you can stitch an approximate panorama
  from the input views. Numbers will be noisier; report in the appendix.

For each example, you need: input views $V_1, V_2$, the question, the
gold answer, and the GT render of every $T$ that applies. Coverage of
the available oracle artifacts on the COSMIC test split (after Steps
1–6 of the renders plan):

| View | Coverage |
|------|----------|
| $T_{\mathrm{td}}$  | 997 / 1000 |
| $T_{\mathrm{pano}}$ | 854 / 1000 (deficit logged in `panorama_missing_per_subtask.json`) |
| $T_{\mathrm{cor}}$ | 498 / 500 (anchor + counting only — N/A on rel-dist / rel-dir per §6) |
| $T_{\mathrm{cot}}$ | populated by the standalone Qwen3-VL annotator; expect ~600–700 retained after wrong-CoT discard |
| $T_{\mathrm{noise}}$ | 1000 / 1000 (deterministic per-`sample_id` assignment from a 100-image pool) |

Provenance index for the eval harness: `two_reader_informativeness_index.json`
(one row per `sample_id` with a path/text per oracle view, or `null`).

## 4. Conditions per question

For every (reader $R$, view type $T$, question $q$):

```
prompt = format_prompt(R, V_1, V_2, T_view, question)
answer = R.generate(prompt)
acc[R, T, q] = exact_match(answer, gold)
```

`T_view` is `None` for the baseline and the GT-rendered view otherwise.
`format_prompt` must be *identical between readers* up to their native
multi-image API, with no helpful hints (e.g. don't say "use the
panorama"). See Section 5 for the prompt template.

Run-list per reader:

- $\mathrm{Acc}_{R,\mathrm{base}}$ (one run, $T_0$)
- $\mathrm{Acc}_{R,T}$ for each $T \in \{T_{\mathrm{cot}}, T_{\mathrm{td}},
  T_{\mathrm{cor}}, T_{\mathrm{pano}}, T_{\mathrm{noise}}\}$

Total: 6 evaluation runs per reader, 12 total. Each is a forward pass over
the eval set; no training.

## 5. Prompt template

Single shared template; only the image slot list changes.

```text
You are answering a spatial reasoning question about an indoor scene.

[IMAGE: View 1]
[IMAGE: View 2]
{IF T != None: [IMAGE/TEXT: extra view of type T]}

Question: {question}
Answer with one of: {answer_options}.
```

Notes:

- Do **not** label the extra view as "panorama" / "top-down" / "ground
  truth" — that leaks information about which condition is running.
- Use the reader's native multi-image format (BAGEL: interleaved
  image+text tokens; Qwen3-VL: chat-style multi-image messages).
- Decoding: greedy, temperature 0, max 32 tokens. Same for both readers.

## 6. Per-T applicability mask

Not every view type is meaningful for every COSMIC subtask. Only score
$T$ on the subtasks where it could plausibly help, otherwise the average
is dragged down by uninformative subtasks (and the reader can get
*confused* by an irrelevant extra image).

| Subtask | $T_{\mathrm{cot}}$ | $T_{\mathrm{td}}$ | $T_{\mathrm{cor}}$ | $T_{\mathrm{pano}}$ |
|---------|:---:|:---:|:---:|:---:|
| Anchor | Y | Y | **Y** | Y |
| Counting | Y | Y | **Y** | Y |
| Rel-Distance | Y | **Y** | – | Y |
| Rel-Direction | Y | Y | – | **Y** |

(Rotation is intentionally excluded: it is not part of COSMIC `approved_mcqs_*_normalized.json`.
A separate Matterport-side experiment can be added later if needed.)

Bold = expected primary beneficiary. "–" = exclude this $T$ on this
subtask.

Report two numbers per $(R, T)$:

- $I_R(T)\;|\;\text{applicable}$: average over subtasks marked Y/bold
  (this is the headline number).
- $I_R(T)\;|\;\text{all}$: average over all subtasks (appendix only,
  for completeness).

## 7. Metrics

For reader $R$ and view type $T$:

- Raw uplift: $I_R(T) = \mathrm{Acc}_{R,T} - \mathrm{Acc}_{R,\mathrm{base}}$
- Headroom-normalized uplift:
  $I_R^{\mathrm{norm}}(T) = \dfrac{\mathrm{Acc}_{R,T} - \mathrm{Acc}_{R,\mathrm{base}}}{1 - \mathrm{Acc}_{R,\mathrm{base}}}$

Use `I_norm` for the cross-reader scatter — raw `I` is not comparable
across readers because Qwen3-VL's $\mathrm{Acc}_{\mathrm{base}}$ will be
much higher than BAGEL's.

Confidence intervals: bootstrap over the test examples, 1000 resamples,
report 95% CIs. With ~250 COSMIC examples per subtask the CIs will be
~$\pm$3–4 points; mention this when interpreting close calls.

## 8. Sanity checks (must pass before any claim)

1. **Noise condition near zero.** $|I_R(T_{\mathrm{noise}})| < $ small
   (say 2 points absolute) for both readers. If a reader gets a
   meaningful uplift from an unrelated image, it is hallucinating from
   prompt structure — investigate before reporting any other condition.
2. **Baseline reproducibility.** Run $\mathrm{Acc}_{R,\mathrm{base}}$
   twice with different prompt orderings (V1,V2 vs V2,V1). Should agree
   within 1–2 points; otherwise prompts are leaking order information.
3. **Per-subtask sign agreement.** For the bold cell of each $T$ in
   Section 6, both readers should give $I_R^{\mathrm{norm}}(T) > 0$. If
   $T$'s primary beneficiary subtask comes out negative for both
   readers, the rendering is broken (visual artifact, alignment off,
   etc.) — check before drawing conclusions.

## 9. Reporting

**Main figure (paper body).**

A 2D scatter:

- x-axis: $I^{\mathrm{norm}}_{\mathrm{BAGEL}}(T)$
- y-axis: $I^{\mathrm{norm}}_{\mathrm{Qwen}}(T)$
- One point per view type $T$; size $\propto$ headroom-normalized
  uplift averaged across readers.
- Include $T_{\mathrm{noise}}$ near the origin as a visual control.
- Diagonal reference line $y=x$.

Caption: state explicitly that "diagonal alignment indicates $T$'s
informativeness is reader-robust."

**Main table (paper body).**

| View type | $\mathrm{Acc}_{\mathrm{BAGEL,base}}$ | $\mathrm{Acc}_{\mathrm{BAGEL},T}$ | $I_{\mathrm{BAGEL}}^{\mathrm{norm}}$ | $\mathrm{Acc}_{\mathrm{Qwen,base}}$ | $\mathrm{Acc}_{\mathrm{Qwen},T}$ | $I_{\mathrm{Qwen}}^{\mathrm{norm}}$ |

One row per $T$, applicable subtasks only.

**Appendix.**

- Per-subtask breakdown for both readers.
- Transport result on the small real-world subset.
- Prompt templates, decoding settings, model versions.

## 10. Implementation checklist

Render-side (DONE — see plan
`two-reader-informativeness-design-md-th-humble-shell.md`):
- [x] $T_{\mathrm{td}}$: render top-down maps for 4 COSMIC subtasks
      (`generate_topdown_approved_mcqs.sh` + reused
      `generate_topdown_from_dataset.py`).
- [x] $T_{\mathrm{pano}}$: existing panoramas; deficit list at
      `panorama_missing_per_subtask.json` (146/1000 missing).
- [x] $T_{\mathrm{cor}}$: 2D-bbox composite from `visible_objects.json`
      (`build_correspondence_pairs.py`) on anchor + counting.
- [ ] $T_{\mathrm{cot}}$: launch
      `annotate_text_reasoning_approved_mcqs.sh` (Qwen3-VL-235B-FP8
      vLLM, 4×A100 short-unkillable).
- [x] $T_{\mathrm{noise}}$: 100-pano pool + per-sample assignment
      (`build_noise_pool.py`).
- [x] Provenance index (`build_two_reader_index.py`).

Eval-side (TODO, separate plan):
- [ ] Extend `run_inference_bagel_spatial.py` to read the provenance
      index and inject `T_view` as a third image (the
      `InterleaveInferencer` loop already supports N>2 images, so the
      change is at the input-list construction site, lines 428/453).
- [ ] Mirror the same logic in
      `eval/qwen/run_inference_qwen_spatial.py` (just append a third
      `image_url` content block).
- [ ] Build the unified prompt template wrapper that takes
      `(reader, V1, V2, T_view_or_None, question)` and dispatches to
      the correct API.
- [ ] Evaluation harness that runs the 6 conditions × 2 readers =
      12 runs, logs raw outputs, computes accuracies.
- [ ] Bootstrap CI script.
- [ ] Plot script for the 2D scatter.

## 11. Compute estimate

- BAGEL inference: ~6 conditions × ~1.5K applicable items × 1s/item
  ≈ 2.5 GPU-hours.
- Qwen3-VL-72B local inference (if available): ~10 GPU-hours; via API,
  cost-bound rather than time-bound (~$50 estimated for 6 × 1.5K × ~3
  image tokens).

Total: well under one day end-to-end after rendering is done. Rendering
is the long pole if correspondence views need a new pipeline.

## 12. What this experiment does NOT measure

- It does **not** measure learnability $L(T)$. That is a separate
  experiment that compares model-generated views vs GT.
- It does **not** measure end-to-end benefit $E(T)$. That is the main
  table (Pan, Top-down, Correspondence rows).
- It does **not** answer "does generation matter beyond access" — that
  is the generation-vs-understanding probe.

Keep the three experiments cleanly separated in the paper.
