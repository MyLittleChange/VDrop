"""
Extended Bagel model class with spatial image token support.

SpatialBagel extends Bagel to support <|spatial_image_start|> / <|spatial_image_end|>
tokens in prepare_vae_latent(). The diffusion pipeline (generate_image, _forward_flow)
is inherited unchanged.
"""

import torch
from modeling.bagel.bagel import Bagel


class SpatialBagel(Bagel):
    """
    Extends Bagel with spatial image token support.

    Usage:
        model = SpatialBagel.from_pretrained(...)  # or wrap existing model
        model.set_spatial_tokens(start_id, end_id)
        # Then use prepare_vae_latent(..., spatial=True) for spatial images
    """

    def __init__(self, config):
        super().__init__(config)
        self._spatial_start_token_id = None
        self._spatial_end_token_id = None

    def set_spatial_tokens(self, start_of_spatial_image, end_of_spatial_image):
        """Set the spatial image token IDs after loading the model."""
        self._spatial_start_token_id = start_of_spatial_image
        self._spatial_end_token_id = end_of_spatial_image

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids, spatial=False):
        """
        Prepare VAE latent tokens for diffusion generation.

        Args:
            curr_kvlens: Current KV cache lengths.
            curr_rope: Current RoPE position IDs.
            image_sizes: List of (H, W) tuples for images to generate.
            new_token_ids: Dict of token IDs.
            spatial: If True, use spatial image tokens as delimiters.

        Returns:
            generation_input: Dict of tensors for the diffusion loop.
        """
        if spatial:
            assert self._spatial_start_token_id is not None, (
                "Spatial tokens not set. Call model.set_spatial_tokens(start_id, end_id) first."
            )
            start_token_id = self._spatial_start_token_id
            end_token_id = self._spatial_end_token_id
        else:
            start_token_id = new_token_ids['start_of_image']
            end_token_id = new_token_ids['end_of_image']

        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(start_token_id)
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_init_noises.append(
                torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2)
            )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(end_token_id)
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input


def wrap_as_spatial_bagel(model):
    """
    Utility to add spatial token support to an existing Bagel model instance
    without re-instantiating from config.

    This monkey-patches the instance with SpatialBagel methods.

    Usage:
        model = Bagel.from_pretrained(...)
        wrap_as_spatial_bagel(model)
        model.set_spatial_tokens(start_id, end_id)
    """
    import types

    model._spatial_start_token_id = None
    model._spatial_end_token_id = None
    model.set_spatial_tokens = types.MethodType(SpatialBagel.set_spatial_tokens, model)
    model.prepare_vae_latent = types.MethodType(SpatialBagel.prepare_vae_latent, model)

    return model
