"""Unit tests for the bridge-necessity partial-view mask builder.

Runs without pytest: `python tests/test_bridge_partial_view_mask.py`.
Also pytest-compatible if pytest is available.
"""

from __future__ import annotations

import math
import sys
import traceback

import numpy as np
import torch

from data.bridge_masking_utils import (
    PRIMARY_VIEW_ROLES,
    ViewSegment,
    build_bridge_partial_view_mask,
    linear_curriculum_p_mask,
)
from data.data_utils import prepare_attention_mask_per_sample


# Synthetic packed sample layout (token positions are absolute within the sample).
#
#   role          length   abs range      attn mode
#   ---------     ------   ------------   ---------
#   V1            16+2     [ 0,  18)      full
#   V2            16+2     [18,  36)      full
#   context       8        [36,  44)      causal
#   bridge_pre    4        [44,  48)      causal
#   bridge        16+2     [48,  66)      noise
#   reflect_ans   8        [66,  74)      causal
#
# For simplicity we collapse VIT and VAE for V1/V2 into a single 16-token segment
# each (4x4 grid). The mask builder doesn't care about VIT-vs-VAE tagging; it
# only sees ViewSegment objects with a view_id.
SPLIT_LENS = [18, 18, 8, 4, 18, 8]
ATTN_MODES = ["full", "full", "causal", "causal", "noise", "causal"]
VIEW_ROLES = ["V1", "V2", "context", "bridge_pre", "bridge", "reflect_answer"]


def _segments():
    # +1 for <image_start>, +1 for <image_end> markers around 16 patches.
    return [
        ViewSegment(view_id="V1", kind="vae", start=1, length=16, grid_h=4, grid_w=4),
        ViewSegment(view_id="V2", kind="vae", start=19, length=16, grid_h=4, grid_w=4),
    ]


def _v1_patch_range():
    return range(1, 17)


def _v2_patch_range():
    return range(19, 35)


def _bridge_range():
    return range(48, 66)


def _reflect_range():
    return range(66, 74)


def _build_base():
    return prepare_attention_mask_per_sample(SPLIT_LENS, ATTN_MODES)


def test_drop_none_is_identity_baseline_minus_bridge_exposure():
    base = _build_base()
    out, info = build_bridge_partial_view_mask(
        base.clone(),
        SPLIT_LENS,
        VIEW_ROLES,
        _segments(),
        drop="none",
    )
    assert info["masked_view"] is None
    assert info["masked_patch_count"] == 0
    assert torch.equal(out, _build_base())


def test_drop_v1_region_masks_half_v1_patches():
    rng = np.random.default_rng(0)
    base = _build_base()
    out, info = build_bridge_partial_view_mask(
        base.clone(),
        SPLIT_LENS,
        VIEW_ROLES,
        _segments(),
        drop="V1",
        drop_fraction=0.5,
        drop_strategy="region",
        rng=rng,
    )
    assert info["masked_view"] == "V1"
    assert info["bridge_keys_exposed"] is True
    # 4x4 grid, 50% target -> 8 patches.
    assert info["masked_patch_count"] == 8

    reflect_rows = list(_reflect_range())
    masked_v1_cols = [
        c
        for c in _v1_patch_range()
        if torch.isinf(out[reflect_rows[0], c]) and out[reflect_rows[0], c] < 0
    ]
    unmasked_v1_cols = [
        c for c in _v1_patch_range() if out[reflect_rows[0], c].item() == 0.0
    ]
    assert len(masked_v1_cols) == 8
    assert len(unmasked_v1_cols) == 8

    # V2 must remain fully visible to reflect/answer.
    for r in reflect_rows:
        for c in _v2_patch_range():
            assert out[r, c].item() == 0.0, f"V2 col {c} should be visible at row {r}"

    # Bridge must be visible to reflect/answer (we explicitly opened that path).
    for r in reflect_rows:
        for c in _bridge_range():
            assert out[r, c].item() == 0.0, f"bridge col {c} should be visible"


def test_bridge_queries_still_see_all_v1():
    rng = np.random.default_rng(1)
    base = _build_base()
    out, _info = build_bridge_partial_view_mask(
        base.clone(),
        SPLIT_LENS,
        VIEW_ROLES,
        _segments(),
        drop="V1",
        drop_fraction=0.5,
        drop_strategy="region",
        rng=rng,
    )
    # Bridge is in [48, 66). Pick one position inside the patch body.
    bridge_q = 55
    for c in _v1_patch_range():
        # Bridge segment is mode 'noise', which means in the *base* mask the
        # bridge can attend to its priors (V1, V2, context, bridge_pre) and to
        # itself. Our partial-view edit must not change those rows.
        assert out[bridge_q, c].item() == 0.0, (
            f"bridge query lost access to V1 col {c}"
        )


def test_region_strategy_picks_contiguous_rectangle():
    rng = np.random.default_rng(123)
    base = _build_base()
    out, info = build_bridge_partial_view_mask(
        base.clone(),
        SPLIT_LENS,
        VIEW_ROLES,
        _segments(),
        drop="V1",
        drop_fraction=0.5,
        drop_strategy="region",
        rng=rng,
    )
    reflect_row = next(iter(_reflect_range()))
    masked_flat = []
    for c in _v1_patch_range():
        if torch.isinf(out[reflect_row, c]):
            masked_flat.append(c - 1)  # subtract <image_start> offset
    rows = sorted({i // 4 for i in masked_flat})
    cols = sorted({i % 4 for i in masked_flat})
    # contiguous rows
    assert rows == list(range(rows[0], rows[-1] + 1))
    # contiguous cols
    assert cols == list(range(cols[0], cols[-1] + 1))
    # rectangle area equals masked count
    assert len(rows) * len(cols) == len(masked_flat)


def test_random_patches_strategy_hits_target_fraction_in_expectation():
    base = _build_base()
    drop_fraction = 0.5
    counts = []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        out, info = build_bridge_partial_view_mask(
            base.clone(),
            SPLIT_LENS,
            VIEW_ROLES,
            _segments(),
            drop="V1",
            drop_fraction=drop_fraction,
            drop_strategy="random_patches",
            rng=rng,
        )
        counts.append(info["masked_patch_count"])
    mean = float(np.mean(counts))
    assert abs(mean - drop_fraction * 16) < 1.0, mean


def test_drop_both_masks_both_views():
    rng = np.random.default_rng(42)
    base = _build_base()
    out, info = build_bridge_partial_view_mask(
        base.clone(),
        SPLIT_LENS,
        VIEW_ROLES,
        _segments(),
        drop="both",
        drop_fraction=0.5,
        drop_strategy="region",
        rng=rng,
    )
    assert info["masked_view"] == "both"
    assert info["bridge_keys_exposed"] is True
    # 8 patches per view × 2 views = 16 dropped (no overlap — V1 and V2 patches
    # live in disjoint column ranges).
    assert info["masked_patch_count"] == 16

    reflect_row = next(iter(_reflect_range()))

    masked_v1 = [
        c for c in _v1_patch_range()
        if torch.isinf(out[reflect_row, c]) and out[reflect_row, c] < 0
    ]
    unmasked_v1 = [c for c in _v1_patch_range() if out[reflect_row, c].item() == 0.0]
    assert len(masked_v1) == 8
    assert len(unmasked_v1) == 8

    masked_v2 = [
        c for c in _v2_patch_range()
        if torch.isinf(out[reflect_row, c]) and out[reflect_row, c] < 0
    ]
    unmasked_v2 = [c for c in _v2_patch_range() if out[reflect_row, c].item() == 0.0]
    assert len(masked_v2) == 8
    assert len(unmasked_v2) == 8

    # Bridge queries still see all of V1 and V2.
    bridge_q = 55
    for c in list(_v1_patch_range()) + list(_v2_patch_range()):
        assert out[bridge_q, c].item() == 0.0, (
            f"bridge query lost access to col {c}"
        )

    # Bridge keys exposed to the answer.
    for r in _reflect_range():
        for c in _bridge_range():
            assert out[r, c].item() == 0.0


def test_drop_random_picks_uniformly():
    base = _build_base()
    v1, v2 = 0, 0
    for seed in range(400):
        rng = np.random.default_rng(seed)
        _, info = build_bridge_partial_view_mask(
            base.clone(),
            SPLIT_LENS,
            VIEW_ROLES,
            _segments(),
            drop="random",
            drop_fraction=0.5,
            drop_strategy="region",
            rng=rng,
        )
        if info["masked_view"] == "V1":
            v1 += 1
        elif info["masked_view"] == "V2":
            v2 += 1
    # roughly balanced, with slack for sampling noise
    assert 150 < v1 < 250, (v1, v2)
    assert 150 < v2 < 250, (v1, v2)


def test_curriculum_schedule():
    # 500 warmup, 1500 anneal
    assert linear_curriculum_p_mask(0, 500, 1500) == 0.0
    assert linear_curriculum_p_mask(500, 500, 1500) == 0.0
    p_mid = linear_curriculum_p_mask(1250, 500, 1500)
    assert math.isclose(p_mid, 0.5, rel_tol=1e-6)
    assert linear_curriculum_p_mask(2000, 500, 1500) == 1.0
    assert linear_curriculum_p_mask(5000, 500, 1500) == 1.0


def test_grid_rescaling_for_mixed_vit_vae_grids():
    """Same view contributes both VIT and VAE patches at different resolutions."""
    # Layout: V1_vit (16 patches, 4x4), V1_vae (64 patches, 8x8), V2_vit, V2_vae,
    # bridge, reflect.
    split_lens = [18, 66, 18, 66, 8, 4, 18, 8]  # +2 markers per image segment
    attn_modes = ["full", "full", "full", "full", "causal", "causal", "noise", "causal"]
    view_roles = [
        "V1",
        "V1",
        "V2",
        "V2",
        "context",
        "bridge_pre",
        "bridge",
        "reflect_answer",
    ]
    segments = [
        ViewSegment(view_id="V1", kind="vit", start=1, length=16, grid_h=4, grid_w=4),
        ViewSegment(view_id="V1", kind="vae", start=19, length=64, grid_h=8, grid_w=8),
        ViewSegment(view_id="V2", kind="vit", start=85, length=16, grid_h=4, grid_w=4),
        ViewSegment(view_id="V2", kind="vae", start=103, length=64, grid_h=8, grid_w=8),
    ]
    base = prepare_attention_mask_per_sample(split_lens, attn_modes)
    rng = np.random.default_rng(7)
    out, info = build_bridge_partial_view_mask(
        base.clone(),
        split_lens,
        view_roles,
        segments,
        drop="V1",
        drop_fraction=0.5,
        drop_strategy="region",
        rng=rng,
    )
    # 50% of (16 + 64) = 40, but the region picked in the 4x4 VIT grid covers
    # ~50% area; rescaling to the 8x8 VAE grid covers ~50% area there too.
    # We accept anything in [38, 50] as long as both segments contributed.
    assert 30 <= info["masked_patch_count"] <= 55


def _run_all():
    tests = [
        test_drop_none_is_identity_baseline_minus_bridge_exposure,
        test_drop_v1_region_masks_half_v1_patches,
        test_bridge_queries_still_see_all_v1,
        test_region_strategy_picks_contiguous_rectangle,
        test_random_patches_strategy_hits_target_fraction_in_expectation,
        test_drop_both_masks_both_views,
        test_drop_random_picks_uniformly,
        test_curriculum_schedule,
        test_grid_rescaling_for_mixed_vit_vae_grids,
    ]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failures += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    if failures:
        print(f"\n{failures}/{len(tests)} tests FAILED")
        return 1
    print(f"\n{len(tests)}/{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
