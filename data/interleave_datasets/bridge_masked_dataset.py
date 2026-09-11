"""Dataset subclass that tags each entry of `sequence_plan` with a view_role.

Reuses the parent's parse_row logic and parquet schema; the only change is
that every plan entry carries a `view_role` field so downstream packing knows
which segments belong to V1, V2, the bridge, or the reflection-and-answer
span. View-role tagging is a pure metadata addition — no behavior change for
downstream code that doesn't read it.

See data/bridge_masking_utils.py for what the role tags are used for.
"""

from __future__ import annotations

import io

from PIL import Image

from .edit_dataset import (
    SpatialReasoningIterableDataset,
    VisualOnlyThinkingIterableDataset,
)
from ..data_utils import pil_img2rgb


PRIMARY_VIEW_TAGS = ("V1", "V2")


def _tag_role(data, start_idx: int, role: str) -> None:
    """Set view_role on every plan entry from start_idx to the end."""
    for entry in data["sequence_plan"][start_idx:]:
        entry["view_role"] = role


class BridgeMaskedSpatialReasoningIterableDataset(SpatialReasoningIterableDataset):
    """Same parquet layout as the parent.

    image_list = [V1, V2(, extra_view_2, extra_view_3, ...), bridge_image]
    instruction_list = [instruction]
    output_text_list = [thought_0 + <image_start>, <image_end> + thought_1 + answer]

    Tags applied:
        - V1                 : every plan entry from images[0]
        - V2                 : every plan entry from images[1]
        - extra_view_<i>     : every plan entry from images[2:-1] (rotation-style)
        - context            : instruction text
        - bridge_pre         : outputs[0] (thought_0 + <image_start>)
        - bridge             : the trailing reasoning image (B)
        - reflect_answer     : outputs[1] (<image_end> + thought_1 + answer)
    """

    def parse_row(self, row):
        data = self._init_data()
        instrs = row["instruction_list"]
        images = row["image_list"]
        outputs = row["output_text_list"]

        # Input views: first two are primary (V1, V2); any extras are tagged
        # extra_view_<idx> and stay fully visible to the answer.
        for view_idx, img_bytes in enumerate(images[:-1]):
            before = len(data["sequence_plan"])
            data = self._add_image(
                data,
                pil_img2rgb(Image.open(io.BytesIO(img_bytes))),
                need_loss=False,
                need_vae=True,
                need_vit=True,
            )
            if view_idx < len(PRIMARY_VIEW_TAGS):
                role = PRIMARY_VIEW_TAGS[view_idx]
            else:
                role = f"extra_view_{view_idx}"
            _tag_role(data, before, role)

        before = len(data["sequence_plan"])
        data = self._add_text(data, instrs[0], need_loss=False)
        _tag_role(data, before, "context")

        before = len(data["sequence_plan"])
        data = self._add_text(data, outputs[0], need_loss=True)
        _tag_role(data, before, "bridge_pre")

        before = len(data["sequence_plan"])
        data = self._add_image(
            data,
            pil_img2rgb(Image.open(io.BytesIO(images[-1]))),
            need_loss=True,
            need_vae=True,
            need_vit=True,
        )
        _tag_role(data, before, "bridge")

        if len(outputs) > 1:
            before = len(data["sequence_plan"])
            data = self._add_text(data, outputs[1], need_loss=True)
            _tag_role(data, before, "reflect_answer")

        return data


# VisualOnly uses the same row layout, so the masked subclass also serves it.
BridgeMaskedVisualOnlyThinkingIterableDataset = (
    BridgeMaskedSpatialReasoningIterableDataset
)
