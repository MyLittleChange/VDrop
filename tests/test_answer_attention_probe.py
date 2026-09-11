"""Unit tests for answer-token attention span extraction.

Runs without pytest: `python tests/test_answer_attention_probe.py`.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from debug_attention import AttentionDebugger, AttentionSpan


def _fake_debugger() -> AttentionDebugger:
    debugger = object.__new__(AttentionDebugger)
    debugger.num_heads = 1
    debugger.num_kv_heads = 1
    debugger.head_dim = 1
    debugger.kv_group_size = 1
    debugger.captured_data = {
        0: {
            0: {
                "q": torch.tensor([[[1.0]]]),
                "k": torch.tensor([[[0.0]], [[10.0]], [[10.0]], [[0.0]]]),
                "cu_seqlens_q": torch.tensor([0, 1], dtype=torch.int32),
                "cu_seqlens_k": torch.tensor([0, 4], dtype=torch.int32),
            }
        }
    }
    return debugger


def test_attention_span_roundtrip_and_lengths():
    spans = [
        AttentionSpan("V1_vae", 1, 17, kind="image_vae", grid_size=(4, 4), source="V1"),
        AttentionSpan("V2_vit", 19, 35, kind="image_vit", grid_size=(4, 4), source="V2"),
        AttentionSpan("bridge_vae", 48, 64, kind="image_vae", grid_size=(4, 4), source="bridge"),
        AttentionSpan("answer_text", 66, 74, kind="text"),
    ]
    assert [span.length for span in spans] == [16, 16, 16, 8]
    restored = [AttentionSpan.from_dict(span.to_dict()) for span in spans]
    assert restored == spans


def test_query_attention_to_span_mass_is_high_for_forced_keys():
    debugger = _fake_debugger()
    bridge = AttentionSpan("bridge_vae", 1, 3, kind="image_vae", grid_size=(1, 2))
    distractor = AttentionSpan("distractor", 0, 1, kind="text")

    bridge_mass = debugger.extract_query_attention_to_span(0, 0, bridge)
    distractor_mass = debugger.extract_query_attention_to_span(0, 0, distractor)

    assert bridge_mass.item() > 0.99
    assert distractor_mass.item() < 0.01


def test_query_to_grid_heatmap_matches_span_grid():
    debugger = _fake_debugger()
    bridge = AttentionSpan("bridge_vae", 1, 3, kind="image_vae", grid_size=(1, 2))

    heatmap = debugger.extract_query_to_grid_heatmap(0, 0, bridge)

    assert tuple(heatmap.shape) == (1, 2)
    assert torch.all(heatmap > 0.49)


TESTS = [
    test_attention_span_roundtrip_and_lengths,
    test_query_attention_to_span_mass_is_high_for_forced_keys,
    test_query_to_grid_heatmap_matches_span_grid,
]


def main() -> None:
    passed = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
    print(f"{passed}/{len(TESTS)} tests passed")
    if passed != len(TESTS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
