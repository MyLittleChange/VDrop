"""
Extended inferencer with spatial image generation support.

SpatialInterleaveInferencer extends InterleaveInferencer to detect
<|spatial_image_start|> in generated text and trigger diffusion with
configurable spatial image resolution.
"""

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any, Tuple

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from inferencer import InterleaveInferencer


SPATIAL_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. For spatial reasoning images, enclose within <|spatial_image_start|> </|spatial_image_end|>, e.g. <|spatial_image_start|> spatial image here </|spatial_image_end|>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''


class SpatialInterleaveInferencer(InterleaveInferencer):
    """
    Extends InterleaveInferencer with spatial image generation support.

    When the model generates text containing '<|spatial_image_start|>', this
    inferencer triggers diffusion-based image generation using a configurable
    spatial image shape (which can differ from regular image_shapes).
    """

    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        super().__init__(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)

    @torch.no_grad()
    def gen_spatial_image(
        self,
        image_shape,
        gen_context,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_text_precontext=None,
        cfg_img_precontext=None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50,
        timestep_shift=3.0,
        enable_taylorseer=False,
        debug_attn_callback=None,
    ):
        """
        Generate a spatial image using the diffusion pipeline.

        Uses prepare_vae_latent(..., spatial=True) to insert spatial tokens
        as delimiters instead of regular vision tokens.
        """
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
            spatial=True,
        )

        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            enable_taylorseer=enable_taylorseer,
            debug_attn_callback=debug_attn_callback,
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        return image

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        spatial_image_shapes=None,
        enable_taylorseer=False,
        max_rounds: int = 3,
        debug_attn_callback=None,
    ) -> List[Union[str, Image.Image, Tuple[str, Image.Image]]]:
        """
        Extended interleave inference that detects both regular and spatial image tokens.

        Args:
            spatial_image_shapes: (H, W) tuple for spatial image generation resolution.
                                  If None, falls back to image_shapes.
            (all other args are identical to parent)

        Returns:
            List of outputs. Regular images are Image.Image objects.
            Spatial images are returned as ('spatial_image', Image.Image) tuples
            so the caller can distinguish them.
        """
        if spatial_image_shapes is None:
            spatial_image_shapes = image_shapes

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    from inferencer import VLM_THINK_SYSTEM_PROMPT
                    system_prompt = VLM_THINK_SYSTEM_PROMPT
                else:
                    system_prompt = SPATIAL_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)

            else:
                rounds = 0
                while rounds < max_rounds:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    output_list.append(gen_text)
                    gen_context = self.update_context_text(gen_text, gen_context)

                    if "<|spatial_image_start|>" in gen_text:
                        # Spatial image generation
                        img = self.gen_spatial_image(
                            spatial_image_shapes,
                            gen_context,
                            cfg_text_precontext=cfg_text_context,
                            cfg_img_precontext=cfg_img_context,
                            cfg_text_scale=cfg_text_scale,
                            cfg_img_scale=cfg_img_scale,
                            cfg_interval=cfg_interval,
                            timestep_shift=timestep_shift,
                            num_timesteps=num_timesteps,
                            cfg_renorm_min=cfg_renorm_min,
                            cfg_renorm_type=cfg_renorm_type,
                            debug_attn_callback=debug_attn_callback,
                        )
                        output_list.append(('spatial_image', img))

                        img_input = self.vae_transform.resize_transform(pil_img2rgb(img))
                        gen_context = self.update_context_image(img_input, gen_context, vae=not understanding_output)
                        rounds += 1

                    elif "<image_start>" in gen_text:
                        # Regular image generation (inherited behavior)
                        img = self.gen_image(
                            image_shapes,
                            gen_context,
                            cfg_text_precontext=cfg_text_context,
                            cfg_img_precontext=cfg_img_context,
                            cfg_text_scale=cfg_text_scale,
                            cfg_img_scale=cfg_img_scale,
                            cfg_interval=cfg_interval,
                            timestep_shift=timestep_shift,
                            num_timesteps=num_timesteps,
                            cfg_renorm_min=cfg_renorm_min,
                            cfg_renorm_type=cfg_renorm_type,
                            debug_attn_callback=debug_attn_callback,
                        )
                        output_list.append(img)

                        img_input = self.vae_transform.resize_transform(pil_img2rgb(img))
                        gen_context = self.update_context_image(img_input, gen_context, vae=not understanding_output)
                        rounds += 1
                    else:
                        break

        return output_list
