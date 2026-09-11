"""
Extended inferencer with dynamic visual mode token support.

DynamicVisualInterleaveInferencer detects which visual thinking mode the model
chose by scanning the generated text for a mode token appearing **before**
<image_start>, then runs diffusion at the corresponding image size using the
standard gen_image() pipeline (no N+3 changes needed).

Token order in generated text:
    <panoramic> <image_start> ... <image_end>   → 720×1024
    <BEV> <image_start> ... <image_end>         → 720×720

If no mode token precedes <image_start>, falls back to the caller-supplied
image_shapes (same as base InterleaveInferencer behavior).
"""

from copy import deepcopy
from typing import List, Optional, Tuple, Union

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from inferencer import InterleaveInferencer
from SpatialUnderstanding.dynamic_visual_tokens import DYNAMIC_VISUAL_MODES


DYNAMIC_VISUAL_THINK_SYSTEM_PROMPT = (
    "Think visually to answer the question. "
    "Output the mode token before the image block. "
    "For panoramic reasoning (wide-angle view): <panoramic> <image_start> ... <image_end>. "
    "For bird's-eye-view spatial reasoning (top-down): <BEV> <image_start> ... <image_end>."
    "Finally conclude with the final answer wrapped in <answer></answer> tags, "
    "i.e. <answer> answer here </answer>."
)


class DynamicVisualInterleaveInferencer(InterleaveInferencer):
    """
    Extends InterleaveInferencer with dynamic visual mode token support.

    When the model generates text containing a mode token immediately before
    '<image_start>', this inferencer runs diffusion at the mode's predefined
    image size. No changes to prepare_vae_latent or generate_image are needed —
    the mode token is a plain text token; the image block is standard N+2.

    If no mode token precedes '<image_start>', falls back to base gen_image()
    with the caller-supplied image_shapes.
    """

    def _detect_mode(self, gen_text):
        """
        Scan gen_text for a mode token appearing just before '<image_start>'.

        Returns:
            (mode_name, image_size) if a known mode token is found, else (None, None).
        """
        if '<image_start>' not in gen_text:
            return None, None
        before_idx = gen_text.index('<image_start>')
        before = gen_text[:before_idx].rstrip()
        for mode_name, size in DYNAMIC_VISUAL_MODES.items():
            if before.endswith(mode_name):
                return mode_name, size
        return None, None

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
        cfg_interval=None,
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
        max_rounds: int = 3,
        debug_attn_callback=None,
    ) -> List[Union[str, Image.Image, Tuple[str, Image.Image]]]:
        """
        Extended interleave inference that detects dynamic visual mode tokens.

        When the model generates a mode token before <image_start>, the image is
        generated at the mode's predefined size. Otherwise falls back to image_shapes.

        Returns:
            List of outputs. Mode-detected images are returned as
            (mode_name, Image.Image) tuples so the caller can identify the mode used.
            Fallback images are returned as plain Image.Image objects.
        """
        if cfg_interval is None:
            cfg_interval = [0.4, 1.0]

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                system_prompt = DYNAMIC_VISUAL_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(
                        input_term, gen_context, vae=not understanding_output
                    )
                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(
                    gen_context, do_sample=do_sample,
                    temperature=text_temperature, max_length=max_think_token_n
                )
                output_list.append(gen_text)

            else:
                rounds = 0
                while rounds < max_rounds:
                    gen_text = self.gen_text(
                        gen_context, do_sample=do_sample,
                        temperature=text_temperature, max_length=max_think_token_n
                    )
                    output_list.append(gen_text)
                    gen_context = self.update_context_text(gen_text, gen_context)

                    mode_name, image_shape = self._detect_mode(gen_text)
                    if mode_name is not None:
                        # Mode token found before <image_start> — generate at mode's size
                        img = self.gen_image(
                            image_shape,
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
                        output_list.append((mode_name, img))

                        img_input = self.vae_transform.resize_transform(pil_img2rgb(img))
                        gen_context = self.update_context_image(
                            img_input, gen_context, vae=not understanding_output
                        )
                        rounds += 1

                    elif '<image_start>' in gen_text:
                        # Fallback: <image_start> without a preceding mode token
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
                        gen_context = self.update_context_image(
                            img_input, gen_context, vae=not understanding_output
                        )
                        rounds += 1

                    else:
                        break

        return output_list
