# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import random
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class UnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        data = self._init_data()
        instrs  = row["instruction_list"]
        images  = row["image_list"]
        outputs = row["output_text_list"]


        data = self._add_image(
            data,
            pil_img2rgb(Image.open(io.BytesIO(images[0]))),
            need_loss=False,
            need_vae=True,
            need_vit=True,
        )
        data = self._add_text(data, instrs[0], need_loss=False)

        for idx, out_txt in enumerate(outputs):
            data = self._add_text(data, out_txt, need_loss=True)

            img_idx = idx + 1
            if img_idx < len(images):
                data = self._add_image(
                    data,
                    pil_img2rgb(Image.open(io.BytesIO(images[img_idx]))),
                    need_loss=True,
                    need_vae=True,
                    need_vit=True,
                )

        return data


class SpatialReasoningIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    """
    Dataset for spatial reasoning with N input images (problem images) and one output image (reasoning image).

    Expected parquet format:
    - image_list: [problem_image_0, ..., problem_image_N-1, reasoning_image]
    - instruction_list: [instruction]
    - output_text_list: [thought_0 + <image_start>, <image_end> + thought_1 + answer]

    Sequence: [img0]...[imgN-1][instruction] -> [thought_0]<image_start>[reasoning_img]<image_end>[thought_1][answer]

    Supports variable number of input images (e.g., 2 for standard BEV/panorama, 4 for rotation wall views).
    The last image is always treated as the reasoning/output image.
    """

    def parse_row(self, row):
        data = self._init_data()
        instrs = row["instruction_list"]
        images = row["image_list"]
        outputs = row["output_text_list"]

        # Add all input images except the last (no loss, used as input)
        # Supports variable number of input images (2 for standard, 4 for rotation)
        for img_bytes in images[:-1]:
            data = self._add_image(
                data,
                pil_img2rgb(Image.open(io.BytesIO(img_bytes))),
                need_loss=False,
                need_vae=True,
                need_vit=True,
            )

        # Add instruction text (no loss)
        data = self._add_text(data, instrs[0], need_loss=False)

        # Add first output text (thought_0 + <image_start>, with loss)
        data = self._add_text(data, outputs[0], need_loss=True)

        # Add reasoning image — always the last image (with loss)
        data = self._add_image(
            data,
            pil_img2rgb(Image.open(io.BytesIO(images[-1]))),
            need_loss=True,
            need_vae=True,
            need_vit=True,
        )

        # Add second output text (<image_end> + thought_1 + answer, with loss)
        if len(outputs) > 1:
            data = self._add_text(data, outputs[1], need_loss=True)

        return data


# VisualOnlyThinking has the same parse_row logic as SpatialReasoning
# (2 input images + 1 reasoning image with interleaved text outputs).
# The difference is only in the data content, not the loading logic.
VisualOnlyThinkingIterableDataset = SpatialReasoningIterableDataset
