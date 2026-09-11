"""Unit tests for `_zero_bridge_kv` (inference-time bridge KV blanking).

Runs without pytest: `python tests/test_inference_bridge_mask.py`.
Also pytest-compatible if pytest is available.
"""

from __future__ import annotations

import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from inferencer import _zero_bridge_kv
from modeling.bagel.qwen2_navit import NaiveCache


# Synthetic KV layout (positions are absolute within the cache).
#
#   region           kv range       contents
#   ---------------  -------------  ------------------------------------------------
#   prefix (sys+V1)  [ 0, 100)      anything pre-bridge
#   bridge VAE seg   [100, 100+34)  [start_of_image, 32 vae tokens, end_of_image]
#   bridge ViT seg   [134, 134+34)  [start_of_image, 32 vit tokens, end_of_image]
#   suffix (none)    n/a            (answer decode appends after kv_end)
#
# So kv_start=100, kv_end=168, num_image_segments=2, seg_len=34.
# Inner image-token slices that should be zeroed:
#   - [101, 133)
#   - [135, 167)
NUM_LAYERS = 4
NUM_HEADS = 8
HEAD_DIM = 16
PREFIX_LEN = 100
SEG_LEN = 34
KV_START = PREFIX_LEN
KV_END = KV_START + 2 * SEG_LEN  # 168
INNER_VAE = (KV_START + 1, KV_START + SEG_LEN - 1)
INNER_VIT = (KV_START + SEG_LEN + 1, KV_END - 1)


def make_cache(total_len: int) -> NaiveCache:
    cache = NaiveCache(NUM_LAYERS)
    for layer in range(NUM_LAYERS):
        cache.key_cache[layer] = torch.full(
            (total_len, NUM_HEADS, HEAD_DIM), float(layer + 1)
        )
        cache.value_cache[layer] = torch.full(
            (total_len, NUM_HEADS, HEAD_DIM), float(-(layer + 1))
        )
    return cache


def test_zeros_inner_image_tokens_only():
    cache = make_cache(KV_END)
    _zero_bridge_kv(cache, KV_START, KV_END, num_image_segments=2)

    for layer in range(NUM_LAYERS):
        k = cache.key_cache[layer]
        v = cache.value_cache[layer]
        for inner_start, inner_end in (INNER_VAE, INNER_VIT):
            assert torch.all(k[inner_start:inner_end] == 0), (
                f"layer {layer}: inner [{inner_start}:{inner_end}) of K not zeroed"
            )
            assert torch.all(v[inner_start:inner_end] == 0), (
                f"layer {layer}: inner [{inner_start}:{inner_end}) of V not zeroed"
            )


def test_preserves_wrapper_tokens():
    cache = make_cache(KV_END)
    _zero_bridge_kv(cache, KV_START, KV_END, num_image_segments=2)

    wrapper_positions = [
        KV_START,                  # vae start_of_image
        KV_START + SEG_LEN - 1,    # vae end_of_image
        KV_START + SEG_LEN,        # vit start_of_image
        KV_END - 1,                # vit end_of_image
    ]
    for layer in range(NUM_LAYERS):
        k = cache.key_cache[layer]
        v = cache.value_cache[layer]
        for pos in wrapper_positions:
            assert torch.all(k[pos] == float(layer + 1)), (
                f"layer {layer}: wrapper K at pos {pos} was modified"
            )
            assert torch.all(v[pos] == float(-(layer + 1))), (
                f"layer {layer}: wrapper V at pos {pos} was modified"
            )


def test_preserves_prefix():
    cache = make_cache(KV_END)
    _zero_bridge_kv(cache, KV_START, KV_END, num_image_segments=2)

    for layer in range(NUM_LAYERS):
        k = cache.key_cache[layer]
        v = cache.value_cache[layer]
        assert torch.all(k[:KV_START] == float(layer + 1)), (
            f"layer {layer}: prefix K was modified"
        )
        assert torch.all(v[:KV_START] == float(-(layer + 1))), (
            f"layer {layer}: prefix V was modified"
        )


def test_single_segment_understanding_mode():
    """understanding_output=True → vit-only → 1 segment, no vae."""
    seg_only_end = KV_START + SEG_LEN
    cache = make_cache(seg_only_end)
    _zero_bridge_kv(cache, KV_START, seg_only_end, num_image_segments=1)

    inner = (KV_START + 1, seg_only_end - 1)
    for layer in range(NUM_LAYERS):
        k = cache.key_cache[layer]
        v = cache.value_cache[layer]
        assert torch.all(k[inner[0]:inner[1]] == 0)
        assert torch.all(v[inner[0]:inner[1]] == 0)
        assert torch.all(k[KV_START] == float(layer + 1))           # start_of_image
        assert torch.all(k[seg_only_end - 1] == float(layer + 1))   # end_of_image


def test_skips_uninitialized_layers():
    """Should not crash on layers with key_cache[l] is None."""
    cache = NaiveCache(NUM_LAYERS)
    cache.key_cache[0] = torch.full((KV_END, NUM_HEADS, HEAD_DIM), 7.0)
    cache.value_cache[0] = torch.full((KV_END, NUM_HEADS, HEAD_DIM), 9.0)
    _zero_bridge_kv(cache, KV_START, KV_END, num_image_segments=2)

    k0 = cache.key_cache[0]
    assert torch.all(k0[INNER_VAE[0]:INNER_VAE[1]] == 0)
    assert torch.all(k0[INNER_VIT[0]:INNER_VIT[1]] == 0)
    assert cache.key_cache[1] is None  # untouched


def test_rejects_indivisible_span():
    cache = make_cache(KV_END + 1)
    try:
        _zero_bridge_kv(cache, KV_START, KV_END + 1, num_image_segments=2)
    except AssertionError:
        return
    raise AssertionError("Expected AssertionError on non-divisible span")


def main() -> int:
    tests = [
        test_zeros_inner_image_tokens_only,
        test_preserves_wrapper_tokens,
        test_preserves_prefix,
        test_single_segment_understanding_mode,
        test_skips_uninitialized_layers,
        test_rejects_indivisible_span,
    ]
    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{passed + failed} tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
