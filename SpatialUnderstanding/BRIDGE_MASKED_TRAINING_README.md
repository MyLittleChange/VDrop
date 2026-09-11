# Bridge-Necessity Training (Partial-View Attention Masking)

## 1. Idea

During training, when the model decodes the reflection text and answer (the
`reflect_answer` span — the `<image_end> + thought_1 + answer` tokens after the
generated bridge image), we **partially hide one of the two input views** from those
query positions via the attention mask. The bridge `B` itself still attends to **both**
input views fully, so all cross-view information has a path to the answer — but only
through `B`. This forces the bridge image to actually encode useful spatial information
rather than being decorative.

We don't drop entire views (too brittle); we mask a **fraction** of the chosen view's
patch tokens as a contiguous rectangular region (MAE-style), with a curriculum that
ramps the masking probability from 0 → 1 over training.

---

## 2. Sequence layout

```
[V1_vae][V1_vit][V2_vae][V2_vit][instruction]
   └── view_role: V1, V1, V2, V2, context

  → [thought_0 + <image_start>]
        └── view_role: bridge_pre

  → [B_vae]
        └── view_role: bridge

  → [<image_end> + thought_1 + answer]
        └── view_role: reflect_answer
```

Each token segment is tagged with a `view_role` so the mask builder can pick out
the rows (queries) and columns (keys) it needs to edit.

---

## 3. Attention mask construction

Starting from BAGEL's standard `prepare_attention_mask_per_sample` output, the
builder does the following per step:

1. **Pick which view to drop** — uniform 50/50 over `{V1, V2}` (single view per step).
2. **Pick a fraction of patch tokens to hide** within that view (default 50%) as a
   **contiguous rectangle** in the view's patch grid.
   - The same fractional rectangle (mapped by spatial overlap) is applied to both
     the VIT and VAE segments of the same view, so the model can't recover the
     hidden region by querying the un-masked stream.
3. **Edit the mask:**
   - For every `reflect_answer` query row × every hidden patch token column → set
     to `-inf`. The answer literally cannot attend to those positions.
   - For every `reflect_answer` query row × every `bridge` key column → set to
     `0.0` (i.e., *open* that path; see §3.1).
4. `<image_start>` / `<image_end>` markers are never masked. Only patch tokens
   are touched.
5. Bridge query rows are never modified — `B` still sees all of `V1` and `V2`.

### 3.1 Opening the bridge → answer path

BAGEL's standard mask tags the bridge image segment with `attn_mode='noise'`,
which sets the bridge's *key columns* to `-inf` for every later query (including
`reflect_answer`). That's correct for image-generation training, but exactly wrong
here: if half of `V1` is hidden and the bridge is also invisible, the answer has
nowhere to recover the missing information from. The mask builder therefore
**explicitly opens** `(reflect_answer_rows × bridge_cols)` to `0.0` whenever
partial-view masking is applied.

### 3.2 VIT / VAE grid rescaling

A single view contributes both a VIT segment (smaller grid, e.g. 4×4 patches) and
a VAE segment (larger grid, e.g. 8×8 patches). To keep the masked region
spatially aligned across them:

- Pick the rectangle in the source grid.
- For each destination grid, map every source patch's spatial rectangle
  `[r/H_src, (r+1)/H_src) × [c/W_src, (c+1)/W_src)` to destination patches whose
  *centres* fall inside it. Nearest-neighbour fallback if the source grid is finer
  than the destination.

---

## 4. Mask ratio and curriculum

**Mask ratio.** Default `drop_fraction = 0.5` — half of the chosen view's patch
tokens are hidden from the answer queries in any masked step.

**Curriculum.** Full attention for the first 500 steps (the model first learns to
generate a coherent panorama), then a linear ramp of the masking probability from
0 → 1 over the next 1500 steps, always-on after step 2000:

```
linear_curriculum_p_mask(step, warmup=500, anneal=1500):
    if step < warmup:                       return 0.0
    if step >= warmup + anneal:             return 1.0
    return (step - warmup) / anneal         # 0.0 → 1.0 linearly
```

Per yielded packed batch we draw one Bernoulli `Uniform(0,1) < p_mask`. If it
fires we apply the partial-view mask edit; otherwise we keep the standard
full-attention mask for that step.
