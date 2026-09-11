"""Partial-view attention masking for the bridge-necessity training scheme.

The model trains on packed sequences shaped roughly as:

    [V1_vae][V1_vit][V2_vae][V2_vit][instruction]
        -> [thought_0 + <image_start>] -> [B_vae] -> [<image_end> + thought_1 + answer]

where V1 / V2 are the two input views, B is the visual bridge (panorama), and
the trailing text span is the reflection + answer. The standard BAGEL mask lets
the answer attend directly to V1 + V2, so the bridge is easy to ignore.

This module builds an attention mask that hides a *fraction* of one input view
(V1 or V2) from the reflection-and-answer queries only. The bridge B still sees
both views fully; the answer must therefore route cross-view information
through B.

Hooks into the existing per-sample float mask returned by
`prepare_attention_mask_per_sample` (see data/data_utils.py) — we never replace
that mask, only zero out additional cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import torch

from .data_utils import prepare_attention_mask_per_sample


# Roles emitted by the bridge-masked dataset; only V1 and V2 are eligible for masking.
PRIMARY_VIEW_ROLES = ("V1", "V2")
REFLECT_ROLE = "reflect_answer"
BRIDGE_ROLE = "bridge"


@dataclass
class ViewSegment:
    """One contiguous segment of patch tokens belonging to a single input view."""

    view_id: str        # "V1" or "V2"
    kind: str           # "vit" or "vae"
    start: int          # absolute position in packed sequence (inclusive)
    length: int         # number of patch tokens
    grid_h: int         # patches along height
    grid_w: int         # patches along width

    @property
    def end(self) -> int:
        return self.start + self.length

    def positions(self) -> np.ndarray:
        return np.arange(self.start, self.end, dtype=np.int64)


def _select_region_indices(
    grid_h: int,
    grid_w: int,
    drop_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick a contiguous axis-aligned rectangle covering ~drop_fraction of patches.

    Returns the *flat* (row-major) patch indices to drop. The rectangle's area
    is the smallest cell count >= drop_fraction * total. Aspect ratio is sampled
    so the rectangle is plausibly shaped (not a single thin line) — we pick a
    height in [1, grid_h] uniformly, then derive width to hit the area target.
    """
    total = grid_h * grid_w
    target = int(math.ceil(drop_fraction * total))
    target = max(1, min(target, total))

    # Sample a height; derive width to hit the area target. Iterate a few times
    # to bias away from degenerate 1xN strips when the grid is large enough.
    best = None
    for _ in range(8):
        h = int(rng.integers(1, grid_h + 1))
        w = int(math.ceil(target / h))
        w = max(1, min(w, grid_w))
        area = h * w
        if best is None or abs(area - target) < abs(best[0] * best[1] - target):
            best = (h, w)
        if area == target:
            break
    h, w = best  # type: ignore[misc]

    top = int(rng.integers(0, grid_h - h + 1))
    left = int(rng.integers(0, grid_w - w + 1))

    rows = np.arange(top, top + h)
    cols = np.arange(left, left + w)
    flat = (rows[:, None] * grid_w + cols[None, :]).reshape(-1)
    return flat


def _select_random_patch_indices(
    grid_h: int,
    grid_w: int,
    drop_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """i.i.d. Bernoulli mask. Returns flat patch indices to drop."""
    total = grid_h * grid_w
    keep = rng.random(total) < drop_fraction
    return np.nonzero(keep)[0]


def _segment_drop_positions(
    seg: ViewSegment,
    drop_fraction: float,
    drop_strategy: str,
    rng: np.random.Generator,
    shared_flat_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick which patch positions inside `seg` to drop.

    If `shared_flat_indices` is provided (the same fractional region picked for
    a sibling segment of the same view, e.g. VIT picked first → VAE reuses
    that region rescaled to its grid), reuse it. Otherwise pick fresh.

    Returns (absolute_positions_dropped, flat_indices_dropped).
    """
    if shared_flat_indices is not None:
        flat = shared_flat_indices
    elif drop_strategy == "region":
        flat = _select_region_indices(seg.grid_h, seg.grid_w, drop_fraction, rng)
    elif drop_strategy == "random_patches":
        flat = _select_random_patch_indices(seg.grid_h, seg.grid_w, drop_fraction, rng)
    else:
        raise ValueError(f"Unknown drop_strategy={drop_strategy!r}")

    flat = np.clip(flat, 0, seg.length - 1).astype(np.int64)
    abs_positions = seg.start + flat
    return abs_positions, flat


def _rescale_flat_indices(
    flat: np.ndarray,
    src_grid: Tuple[int, int],
    dst_grid: Tuple[int, int],
) -> np.ndarray:
    """Rescale a set of flat row-major patch indices from one grid to another.

    Maps the *spatial fraction* covered by the source patches to the
    corresponding patches in the destination grid. Each source patch covers a
    rectangle in [0,1]^2; we hit every destination patch whose centre falls
    inside the union of those rectangles.

    Used so the VIT region we picked maps to the *same fractional rectangle*
    in the VAE grid (different patch size, different grid resolution).
    """
    src_h, src_w = src_grid
    dst_h, dst_w = dst_grid
    if src_h == 0 or src_w == 0 or dst_h == 0 or dst_w == 0 or flat.size == 0:
        return np.array([], dtype=np.int64)

    src_rows = flat // src_w
    src_cols = flat % src_w

    # Source patch (r, c) covers the rectangle [r/src_h, (r+1)/src_h) x
    # [c/src_w, (c+1)/src_w) in the unit square.
    row_lo = src_rows / src_h
    row_hi = (src_rows + 1) / src_h
    col_lo = src_cols / src_w
    col_hi = (src_cols + 1) / src_w

    # Destination patch indices whose centre falls inside the union.
    dst_row_centres = (np.arange(dst_h) + 0.5) / dst_h
    dst_col_centres = (np.arange(dst_w) + 0.5) / dst_w

    dst_indices: List[int] = []
    for rl, rh, cl, ch in zip(row_lo, row_hi, col_lo, col_hi):
        rs = np.where((dst_row_centres >= rl) & (dst_row_centres < rh))[0]
        cs = np.where((dst_col_centres >= cl) & (dst_col_centres < ch))[0]
        if rs.size == 0 or cs.size == 0:
            # Source patch smaller than dst resolution — fall back to nearest.
            r = int(np.clip(np.floor((rl + rh) / 2 * dst_h), 0, dst_h - 1))
            c = int(np.clip(np.floor((cl + ch) / 2 * dst_w), 0, dst_w - 1))
            dst_indices.append(r * dst_w + c)
        else:
            for r in rs:
                for c in cs:
                    dst_indices.append(int(r) * dst_w + int(c))
    return np.unique(np.array(dst_indices, dtype=np.int64))


def _reflect_answer_query_positions(
    split_lens: Sequence[int],
    view_roles: Sequence[str],
) -> np.ndarray:
    """Absolute query positions belonging to the reflection-and-answer span(s)."""
    out: List[int] = []
    cursor = 0
    for length, role in zip(split_lens, view_roles):
        if role == REFLECT_ROLE:
            out.extend(range(cursor, cursor + length))
        cursor += length
    return np.array(out, dtype=np.int64)


def _split_position_ranges(split_lens: Sequence[int]) -> List[Tuple[int, int]]:
    """Per-split (start, end_exclusive) absolute positions within the sample."""
    ranges: List[Tuple[int, int]] = []
    cursor = 0
    for length in split_lens:
        ranges.append((cursor, cursor + length))
        cursor += length
    return ranges


def _bridge_key_positions(
    split_lens: Sequence[int], view_roles: Sequence[str]
) -> np.ndarray:
    """Absolute positions of every token in any 'bridge' segment."""
    out: List[int] = []
    for (start, end), role in zip(_split_position_ranges(split_lens), view_roles):
        if role == BRIDGE_ROLE:
            out.extend(range(start, end))
    return np.array(out, dtype=np.int64)


def _pick_drop_positions_for_view(
    segs_for_view: Sequence[ViewSegment],
    *,
    drop_fraction: float,
    drop_strategy: str,
    rng: np.random.Generator,
) -> List[int]:
    """Pick a fractional region in one view's primary grid and rescale to siblings.

    All segments of the same view (e.g. its VIT and VAE patch streams) share the
    same fractional spatial rectangle so the masked region is consistent across
    them. Returns absolute (per-sample) token positions to drop.
    """
    if not segs_for_view:
        return []

    primary = segs_for_view[0]
    if drop_strategy == "region":
        primary_flat = _select_region_indices(
            primary.grid_h, primary.grid_w, drop_fraction, rng
        )
    elif drop_strategy == "random_patches":
        primary_flat = _select_random_patch_indices(
            primary.grid_h, primary.grid_w, drop_fraction, rng
        )
    else:
        raise ValueError(f"Unknown drop_strategy={drop_strategy!r}")

    out: List[int] = []
    for seg in segs_for_view:
        if (seg.grid_h, seg.grid_w) == (primary.grid_h, primary.grid_w):
            seg_flat = primary_flat
        else:
            seg_flat = _rescale_flat_indices(
                primary_flat,
                (primary.grid_h, primary.grid_w),
                (seg.grid_h, seg.grid_w),
            )
        seg_flat = np.clip(seg_flat, 0, seg.length - 1).astype(np.int64)
        out.extend((seg.start + seg_flat).tolist())
    return out


def build_bridge_partial_view_mask(
    base_mask: torch.Tensor,
    split_lens: Sequence[int],
    view_roles: Sequence[str],
    view_segments: Sequence[ViewSegment],
    *,
    drop: str = "random",
    drop_fraction: float = 0.5,
    drop_strategy: str = "region",
    rng: Optional[np.random.Generator] = None,
    expose_bridge_to_reflect: bool = True,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Apply partial-view masking on top of an existing per-sample mask.

    Args:
        base_mask: float tensor (sample_len, sample_len) from
            `prepare_attention_mask_per_sample`. -inf at masked positions, 0
            otherwise. Modified in place (for memory) and also returned.
        split_lens: per-segment lengths used to build base_mask.
        view_roles: per-segment role tags, len == len(split_lens). Must contain
            'reflect_answer' for the answer/reflection span(s).
        view_segments: list of ViewSegment, one per (view, kind) input image.
            Only segments with view_id in {V1, V2} are eligible for masking.
        drop: 'V1' | 'V2' | 'both' | 'random' | 'random_or_both' | 'none'.
            - 'random' picks one of {V1, V2} uniformly (single view masked).
            - 'both' masks both V1 and V2 simultaneously, each with an
              independently-sampled partial region/patch set.
            - 'random_or_both' picks uniformly from {V1, V2, both}.
            - 'none' returns base_mask unchanged.
            If only one of V1/V2 is present, 'random'/'both'/'random_or_both'
            fall back to whichever is available.
        drop_fraction: fraction of patch tokens to hide (0 < p < 1).
        drop_strategy: 'region' (contiguous rectangle, default) or
            'random_patches' (i.i.d. Bernoulli).
        rng: numpy RNG; if None, a fresh `default_rng()` is used.
        expose_bridge_to_reflect: if True (default), zero out the -inf cells
            on (reflect_answer rows, bridge cols). The base mask treats the
            bridge segment as 'noise' so other segments can't attend to it,
            but for bridge-necessity training we WANT R, Y to attend to B.
            This is the whole point of the curriculum.

    Returns:
        (mask, info) where info is a dict with 'masked_view',
        'masked_patch_count', 'drop_strategy', 'drop_fraction',
        'bridge_keys_exposed'.
    """
    if drop == "none":
        return base_mask, {
            "masked_view": None,
            "masked_patch_count": 0,
            "drop_strategy": drop_strategy,
            "drop_fraction": drop_fraction,
            "bridge_keys_exposed": False,
        }

    if rng is None:
        rng = np.random.default_rng()

    # Step 1 — expose bridge keys to reflect/answer queries. The standard
    # mask hides bridge as 'noise'; we re-open that path so the answer can
    # actually look at B. Without this, the partial-view mask just removes
    # information without offering an alternative route.
    bridge_kv = _bridge_key_positions(split_lens, view_roles)
    reflect_q_all = _reflect_answer_query_positions(split_lens, view_roles)
    bridge_keys_exposed = False
    if expose_bridge_to_reflect and bridge_kv.size > 0 and reflect_q_all.size > 0:
        bridge_kv_t = torch.as_tensor(bridge_kv, dtype=torch.long)
        reflect_q_t = torch.as_tensor(reflect_q_all, dtype=torch.long)
        base_mask[reflect_q_t.unsqueeze(1), bridge_kv_t.unsqueeze(0)] = 0.0
        bridge_keys_exposed = True

    primary_segments_by_view: Dict[str, List[ViewSegment]] = {"V1": [], "V2": []}
    for seg in view_segments:
        if seg.view_id in primary_segments_by_view:
            primary_segments_by_view[seg.view_id].append(seg)

    available = [v for v in PRIMARY_VIEW_ROLES if primary_segments_by_view[v]]
    if not available:
        return base_mask, {
            "masked_view": None,
            "masked_patch_count": 0,
            "drop_strategy": drop_strategy,
            "drop_fraction": drop_fraction,
            "bridge_keys_exposed": bridge_keys_exposed,
        }

    if drop == "random":
        views_to_mask = [available[int(rng.integers(0, len(available)))]]
    elif drop == "both":
        views_to_mask = list(available)
    elif drop in PRIMARY_VIEW_ROLES:
        if drop not in available:
            return base_mask, {
                "masked_view": None,
                "masked_patch_count": 0,
                "drop_strategy": drop_strategy,
                "drop_fraction": drop_fraction,
                "bridge_keys_exposed": bridge_keys_exposed,
            }
        views_to_mask = [drop]
    else:
        raise ValueError(f"Unknown drop={drop!r}")

    if drop == "both" and len(views_to_mask) == 2:
        masked_view_label: object = "both"
    else:
        masked_view_label = views_to_mask[0]

    reflect_q = reflect_q_all
    if reflect_q.size == 0:
        return base_mask, {
            "masked_view": masked_view_label,
            "masked_patch_count": 0,
            "drop_strategy": drop_strategy,
            "drop_fraction": drop_fraction,
            "bridge_keys_exposed": bridge_keys_exposed,
        }

    drop_positions: List[int] = []
    for view in views_to_mask:
        drop_positions.extend(
            _pick_drop_positions_for_view(
                primary_segments_by_view[view],
                drop_fraction=drop_fraction,
                drop_strategy=drop_strategy,
                rng=rng,
            )
        )

    if not drop_positions:
        return base_mask, {
            "masked_view": masked_view_label,
            "masked_patch_count": 0,
            "drop_strategy": drop_strategy,
            "drop_fraction": drop_fraction,
            "bridge_keys_exposed": bridge_keys_exposed,
        }

    drop_kv = torch.as_tensor(sorted(set(drop_positions)), dtype=torch.long)
    reflect_q_t = torch.as_tensor(reflect_q, dtype=torch.long)

    # Mask the (reflect_q × drop_kv) submatrix. base_mask uses 0 for "attend"
    # and -inf for "masked" (see prepare_attention_mask_per_sample).
    base_mask[reflect_q_t.unsqueeze(1), drop_kv.unsqueeze(0)] = float("-inf")

    return base_mask, {
        "masked_view": masked_view_label,
        "masked_patch_count": int(drop_kv.numel()),
        "drop_strategy": drop_strategy,
        "drop_fraction": drop_fraction,
        "bridge_keys_exposed": bridge_keys_exposed,
    }


def linear_curriculum_p_mask(
    step: int, warmup_steps: int, anneal_steps: int
) -> float:
    """Linearly anneal partial-mask probability from 0 to 1 after a warmup."""
    if step < warmup_steps:
        return 0.0
    if anneal_steps <= 0:
        return 1.0
    progress = (step - warmup_steps) / float(anneal_steps)
    return float(min(1.0, max(0.0, progress)))
