"""Subclass of PackedDataset that injects partial-view attention masking.

Behavior:
    - Walks the sample's `sequence_plan` like the parent's pack_sequence, but
      additionally tracks per-split `view_role` and, for input-view images,
      records `ViewSegment`s (start/length/grid in the packed sequence).
    - After the standard mask is built, with probability p_mask (set by a
      linear curriculum on the global step) we hide a fraction of one input
      view's patches from the reflection-and-answer queries via
      `build_bridge_partial_view_mask`.

Designed to be plug-compatible with PackedDataset — the trainer can swap the
class via `pretrain_unified_navit.PackedDataset = BridgeMaskedPackedDataset`
without further changes (see train/pretrain_unified_navit_bridge_masked.py).
"""

from __future__ import annotations

import os
import random
from typing import List, Optional

import numpy as np
import torch

from .data_utils import (
    len2weight,
    patchify,
    prepare_attention_mask_per_sample,
)
from .dataset_base import PackedDataset
from .bridge_masking_utils import (
    PRIMARY_VIEW_ROLES,
    BRIDGE_ROLE,
    REFLECT_ROLE,
    ViewSegment,
    build_bridge_partial_view_mask,
    linear_curriculum_p_mask,
)


# Roles emitted by BridgeMaskedSpatialReasoningIterableDataset that must NEVER
# be masked (they are queries we want to mask FROM, or content we want
# preserved as keys for everyone).
NEVER_MASK_ROLES = {"context", "bridge_pre", BRIDGE_ROLE, REFLECT_ROLE}


class BridgeMaskedPackedDataset(PackedDataset):
    """PackedDataset variant that applies bridge-necessity partial-view masking.

    Extra knobs (read from env vars to avoid touching the trainer's argparse):

        BRIDGE_MASK_WARMUP_STEPS   (default 500)
        BRIDGE_MASK_ANNEAL_STEPS   (default 1500)
        BRIDGE_MASK_DROP_FRACTION  (default 0.5)
        BRIDGE_MASK_DROP_STRATEGY  (default "region", or "random_patches")
        BRIDGE_MASK_DROP           (default "random"; or "V1"/"V2"/"both"/"none")

    Step counter is per-rank-per-worker. Since masking is a per-sample
    decision, drift across workers is harmless — the curriculum is a soft
    schedule, not a synchronization point.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_flex:
            raise NotImplementedError(
                "BridgeMaskedPackedDataset only supports use_flex=False; "
                "the FLEX-attention path uses a different mask builder."
            )
        self._step = 0
        self.bridge_mask_warmup_steps = int(
            os.environ.get("BRIDGE_MASK_WARMUP_STEPS", 500)
        )
        self.bridge_mask_anneal_steps = int(
            os.environ.get("BRIDGE_MASK_ANNEAL_STEPS", 1500)
        )
        self.bridge_mask_drop_fraction = float(
            os.environ.get("BRIDGE_MASK_DROP_FRACTION", 0.5)
        )
        self.bridge_mask_drop_strategy = os.environ.get(
            "BRIDGE_MASK_DROP_STRATEGY", "region"
        )
        self.bridge_mask_drop = os.environ.get("BRIDGE_MASK_DROP", "random")
        self._mask_rng = np.random.default_rng()

    def _curriculum_p(self) -> float:
        return linear_curriculum_p_mask(
            self._step,
            self.bridge_mask_warmup_steps,
            self.bridge_mask_anneal_steps,
        )

    def __iter__(self):
        # The base iterator yields packed samples; bump our step counter once
        # per yielded batch so the curriculum advances independently of the
        # trainer's optimizer step (the two are equal up to gradient
        # accumulation, which we don't use here).
        for data in super().__iter__():
            self._step += 1
            yield data

    def pack_sequence(self, sample, sequence_status):
        """Re-implemented from PackedDataset.pack_sequence with view tracking.

        Behavioral parity with the parent for the standard-mask code path is
        the contract. The only additions are:
            * per-split view_role accumulator
            * ViewSegment list for V1/V2 patch tokens
            * post-build call to build_bridge_partial_view_mask
        """
        image_tensor_list = sample["image_tensor_list"]
        text_ids_list = sample["text_ids_list"]
        sequence_plan = sample["sequence_plan"]

        split_lens: List[int] = []
        attn_modes: List[str] = []
        view_roles: List[str] = []
        view_segments: List[ViewSegment] = []
        curr = sequence_status["curr"]
        curr_rope_id = 0
        sample_lens = 0
        sample_start = curr

        for item in sequence_plan:
            split_start = item.get("split_start", True)
            if split_start:
                curr_split_len = 0
                # Active role for this split is whatever the *first* item that
                # opens it carries. Splits that span multiple plan items
                # (videos) will carry the role of the first item.
                current_role = item.get("view_role", "other")

            if item["type"] == "text":
                text_ids = text_ids_list.pop(0)
                if (
                    item["enable_cfg"] == 1
                    and random.random() < self.data_config.text_cond_dropout_prob
                ):
                    continue

                shifted_text_ids = [self.bos_token_id] + text_ids
                sequence_status["packed_text_ids"].extend(shifted_text_ids)
                sequence_status["packed_text_indexes"].extend(
                    range(curr, curr + len(shifted_text_ids))
                )
                if item["loss"] == 1:
                    sequence_status["ce_loss_indexes"].extend(
                        range(curr, curr + len(shifted_text_ids))
                    )
                    sequence_status["ce_loss_weights"].extend(
                        [len2weight(len(shifted_text_ids))] * len(shifted_text_ids)
                    )
                    sequence_status["packed_label_ids"].extend(
                        text_ids + [self.eos_token_id]
                    )
                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                # add a <|im_end|> token
                sequence_status["packed_text_ids"].append(self.eos_token_id)
                sequence_status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:
                    sequence_status["ce_loss_indexes"].append(curr)
                    sequence_status["ce_loss_weights"].append(1.0)
                    sequence_status["packed_label_ids"].append(
                        item["special_token_label"]
                    )
                curr += 1
                curr_split_len += 1

                attn_modes.append("causal")
                sequence_status["packed_position_ids"].extend(
                    range(curr_rope_id, curr_rope_id + curr_split_len)
                )
                curr_rope_id += curr_split_len

            elif item["type"] == "vit_image":
                image_tensor = image_tensor_list.pop(0)
                if (
                    item["enable_cfg"] == 1
                    and random.random() < self.data_config.vit_cond_dropout_prob
                ):
                    curr_rope_id += 1
                    continue

                sequence_status["packed_text_ids"].append(self.start_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                curr += 1
                curr_split_len += 1

                vit_tokens = patchify(image_tensor, self.data_config.vit_patch_size)
                num_img_tokens = vit_tokens.shape[0]
                seg_start = curr
                sequence_status["packed_vit_token_indexes"].extend(
                    range(curr, curr + num_img_tokens)
                )
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status["packed_vit_tokens"].append(vit_tokens)
                sequence_status["vit_token_seqlens"].append(num_img_tokens)
                sequence_status["packed_vit_position_ids"].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1),
                        image_tensor.size(2),
                        self.data_config.vit_patch_size,
                        max_num_patches_per_side=self.data_config.max_num_patch_per_side,
                    )
                )

                if current_role in PRIMARY_VIEW_ROLES:
                    grid_h = image_tensor.size(1) // self.data_config.vit_patch_size
                    grid_w = image_tensor.size(2) // self.data_config.vit_patch_size
                    view_segments.append(
                        ViewSegment(
                            view_id=current_role,
                            kind="vit",
                            start=seg_start,
                            length=num_img_tokens,
                            grid_h=grid_h,
                            grid_w=grid_w,
                        )
                    )

                sequence_status["packed_text_ids"].append(self.end_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:
                    sequence_status["ce_loss_indexes"].append(curr)
                    sequence_status["ce_loss_weights"].append(1.0)
                    sequence_status["packed_label_ids"].append(
                        item["special_token_label"]
                    )
                curr += 1
                curr_split_len += 1

                attn_modes.append("full")
                sequence_status["packed_position_ids"].extend(
                    [curr_rope_id] * curr_split_len
                )
                curr_rope_id += 1

            elif item["type"] == "vae_image":
                image_tensor = image_tensor_list.pop(0)
                if (
                    item["enable_cfg"] == 1
                    and random.random() < self.data_config.vae_cond_dropout_prob
                ):
                    curr_rope_id += 1
                    continue

                sequence_status["packed_text_ids"].append(self.start_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                curr += 1
                curr_split_len += 1

                sequence_status["vae_image_tensors"].append(image_tensor)
                sequence_status["packed_latent_position_ids"].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1),
                        image_tensor.size(2),
                        self.data_config.vae_image_downsample,
                        max_num_patches_per_side=self.data_config.max_latent_size,
                    )
                )
                H, W = image_tensor.shape[1:]
                h = H // self.data_config.vae_image_downsample
                w = W // self.data_config.vae_image_downsample
                sequence_status["vae_latent_shapes"].append((h, w))

                num_img_tokens = w * h
                seg_start = curr
                sequence_status["packed_vae_token_indexes"].extend(
                    range(curr, curr + num_img_tokens)
                )
                if item["loss"] == 1:
                    sequence_status["mse_loss_indexes"].extend(
                        range(curr, curr + num_img_tokens)
                    )
                    if split_start:
                        timestep = np.random.randn()
                else:
                    timestep = float("-inf")

                sequence_status["packed_timesteps"].extend([timestep] * num_img_tokens)
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                if current_role in PRIMARY_VIEW_ROLES:
                    view_segments.append(
                        ViewSegment(
                            view_id=current_role,
                            kind="vae",
                            start=seg_start,
                            length=num_img_tokens,
                            grid_h=h,
                            grid_w=w,
                        )
                    )

                sequence_status["packed_text_ids"].append(self.end_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:
                    sequence_status["ce_loss_indexes"].append(curr)
                    sequence_status["ce_loss_weights"].append(1.0)
                    sequence_status["packed_label_ids"].append(
                        item["special_token_label"]
                    )
                curr += 1
                curr_split_len += 1

                if split_start:
                    if item["loss"] == 1 and "frame_delta" not in item.keys():
                        attn_modes.append("noise")
                    else:
                        attn_modes.append("full")
                sequence_status["packed_position_ids"].extend(
                    [curr_rope_id] * (num_img_tokens + 2)
                )
                if "frame_delta" in item.keys():
                    curr_rope_id += item["frame_delta"]
                elif item["loss"] == 0:
                    curr_rope_id += 1

            if item.get("split_end", True):
                split_lens.append(curr_split_len)
                view_roles.append(current_role)
                sample_lens += curr_split_len

        sequence_status["curr"] = curr
        sequence_status["sample_lens"].append(sample_lens)

        # Translate absolute (packed-stream) positions to per-sample positions
        # because prepare_attention_mask_per_sample returns a (sample_len,
        # sample_len) matrix indexed from 0.
        for seg in view_segments:
            seg.start -= sample_start

        base_mask = prepare_attention_mask_per_sample(split_lens, attn_modes)

        p_mask = self._curriculum_p()
        if p_mask > 0.0 and self._mask_rng.random() < p_mask:
            mask, _info = build_bridge_partial_view_mask(
                base_mask=base_mask,
                split_lens=split_lens,
                view_roles=view_roles,
                view_segments=view_segments,
                drop=self.bridge_mask_drop,
                drop_fraction=self.bridge_mask_drop_fraction,
                drop_strategy=self.bridge_mask_drop_strategy,
                rng=self._mask_rng,
            )
        else:
            mask = base_mask

        sequence_status["nested_attention_masks"].append(mask)

        return sequence_status
