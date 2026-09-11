"""
Extended dataset classes that support spatial image tokens.

SpatialPackedDataset: Extends PackedDataset to handle 'spatial_vae_image' sequence plan type,
    using <|spatial_image_start|> / <|spatial_image_end|> as delimiters.

SpatialInterleavedBaseIterableDataset: Extends InterleavedBaseIterableDataset with
    _add_spatial_image() helper for building training data with spatial images.
"""

import random
import numpy as np

from data.dataset_base import PackedDataset
from data.data_utils import len2weight
from data.interleave_datasets.interleave_t2i_dataset import InterleavedBaseIterableDataset


class SpatialPackedDataset(PackedDataset):
    """
    Extends PackedDataset to handle 'spatial_vae_image' entries in sequence_plan.

    Spatial VAE images use start_of_spatial_image / end_of_spatial_image tokens
    as delimiters instead of start_of_image / end_of_image.

    All other packing logic (latent tokens, loss indexes, attention modes) is identical.
    """

    def pack_sequence(self, sample, sequence_status):
        image_tensor_list = sample["image_tensor_list"]
        text_ids_list = sample["text_ids_list"]
        sequence_plan = sample["sequence_plan"]

        split_lens, attn_modes = list(), list()
        curr = sequence_status["curr"]
        curr_rope_id = 0
        sample_lens = 0

        for item in sequence_plan:
            split_start = item.get("split_start", True)
            if split_start:
                curr_split_len = 0

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

                # update sequence status
                attn_modes.append("causal")
                sequence_status["packed_position_ids"].extend(
                    range(curr_rope_id, curr_rope_id + curr_split_len)
                )
                curr_rope_id += curr_split_len

            elif item["type"] == "vit_image":
                # Identical to parent — delegate to parent's logic
                image_tensor = image_tensor_list.pop(0)
                if (
                    item["enable_cfg"] == 1
                    and random.random() < self.data_config.vit_cond_dropout_prob
                ):
                    curr_rope_id += 1
                    continue

                from data.data_utils import patchify

                sequence_status["packed_text_ids"].append(self.start_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                curr += 1
                curr_split_len += 1

                vit_tokens = patchify(image_tensor, self.data_config.vit_patch_size)
                num_img_tokens = vit_tokens.shape[0]
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

            elif item["type"] in ("vae_image", "spatial_vae_image"):
                # Determine which delimiter tokens to use
                is_spatial = (item["type"] == "spatial_vae_image")
                if is_spatial:
                    start_token = self.start_of_spatial_image
                    end_token = self.end_of_spatial_image
                else:
                    start_token = self.start_of_image
                    end_token = self.end_of_image

                image_tensor = image_tensor_list.pop(0)
                if (
                    item["enable_cfg"] == 1
                    and random.random() < self.data_config.vae_cond_dropout_prob
                ):
                    curr_rope_id += 1
                    continue

                # add start token
                sequence_status["packed_text_ids"].append(start_token)
                sequence_status["packed_text_indexes"].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
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

                # add end token
                sequence_status["packed_text_ids"].append(end_token)
                sequence_status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:
                    sequence_status["ce_loss_indexes"].append(curr)
                    sequence_status["ce_loss_weights"].append(1.0)
                    sequence_status["packed_label_ids"].append(
                        item["special_token_label"]
                    )
                curr += 1
                curr_split_len += 1

                # update sequence status
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
                sample_lens += curr_split_len

        sequence_status["curr"] = curr
        sequence_status["sample_lens"].append(sample_lens)
        # prepare attention mask
        if not self.use_flex:
            from data.data_utils import prepare_attention_mask_per_sample
            sequence_status["nested_attention_masks"].append(
                prepare_attention_mask_per_sample(split_lens, attn_modes)
            )
        else:
            sequence_status["split_lens"].extend(split_lens)
            sequence_status["attn_modes"].extend(attn_modes)

        return sequence_status


class SpatialInterleavedBaseIterableDataset(InterleavedBaseIterableDataset):
    """
    Extends InterleavedBaseIterableDataset with _add_spatial_image() helper.

    Use this as a base class for datasets that produce spatial image entries
    in their training data.
    """

    def _add_spatial_image(self, data, image, need_loss, need_vae, need_vit, enable_cfg=True):
        """
        Add a spatial image to the training data.

        Similar to _add_image(), but uses 'spatial_vae_image' type instead of 'vae_image'
        for VAE entries. ViT entries remain 'vit_image' (spatial tokens only affect the
        generation/VAE pathway).
        """
        assert need_loss or need_vae or need_vit

        if need_loss:
            data['sequence_plan'].append(
                {
                    'type': 'spatial_vae_image',
                    'enable_cfg': 0,
                    'loss': 1,
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor)

        if need_vae:
            data['sequence_plan'].append(
                {
                    'type': 'spatial_vae_image',
                    'enable_cfg': int(enable_cfg),
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor.clone())

        if need_vit:
            # ViT entries are always 'vit_image' — spatial tokens don't affect understanding
            data['sequence_plan'].append(
                {
                    'type': 'vit_image',
                    'enable_cfg': int(enable_cfg),
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                },
            )
            vit_image_tensor = self.vit_transform(image)
            height, width = vit_image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.vit_transform.stride ** 2
            data['image_tensor_list'].append(vit_image_tensor)

        return data
