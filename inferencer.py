# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache



def _zero_bridge_kv(past_key_values, kv_start: int, kv_end: int, num_image_segments: int = 2):
    """Zero out the KV cache slice for a freshly-appended bridge image.

    `update_context_image(..., vae=True, vit=True)` appends two segments to the
    cache, each laid out as [start_of_image_token, image_tokens..., end_of_image_token].
    For each segment, only the inner image-token rows are zeroed; the two
    structural wrapper text tokens are left intact so the answer decoder still
    sees the `<image_start>`/`<image_end>` markers.

    Args:
        past_key_values: NaiveCache with key_cache[layer]/value_cache[layer]
            tensors of shape [total_kv_len, num_kv_heads, head_dim].
        kv_start: First KV index of the appended bridge region (inclusive).
        kv_end: KV index immediately after the appended bridge region (exclusive).
        num_image_segments: 1 (vit-only, understanding_output=True) or 2 (vae+vit).
    """
    span = kv_end - kv_start
    assert span > 0 and num_image_segments in (1, 2), (
        f"Bad bridge KV span: kv_start={kv_start}, kv_end={kv_end}, "
        f"num_image_segments={num_image_segments}"
    )
    seg_len = span // num_image_segments
    assert seg_len * num_image_segments == span, (
        f"Bridge KV span {span} not divisible into {num_image_segments} equal segments"
    )

    inner_slices = []
    for s in range(num_image_segments):
        seg_start = kv_start + s * seg_len
        seg_end = seg_start + seg_len
        inner_start = seg_start + 1
        inner_end = seg_end - 1
        if inner_end > inner_start:
            inner_slices.append((inner_start, inner_end))

    for layer_idx in range(past_key_values.num_layers):
        k = past_key_values.key_cache[layer_idx]
        v = past_key_values.value_cache[layer_idx]
        if k is None or v is None:
            continue
        for inner_start, inner_end in inner_slices:
            k[inner_start:inner_end] = 0
            v[inner_start:inner_end] = 0


VLM_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''

GEN_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def _append_raw_token_ids(self, token_ids, gen_context):
        """Append raw token IDs to the KV cache *without* bos/eos chat-turn wrapping.

        `update_context_text` always wraps every prompt with `<|im_start|>...<|im_end|>`
        via `prepare_prompts`, which is correct for user/assistant text turns but
        breaks the model's expected `<image_start>...<image_end>` flow when we want
        to inject only the structural image wrapper. This helper feeds arbitrary
        token IDs straight into `forward_cache_update_text`, used by the
        `force_no_visual_thinking` branch to emulate the exact wrapper-token state
        the model saw during training.
        """
        assert isinstance(token_ids, (list, tuple)) and len(token_ids) > 0
        past_key_values = gen_context['past_key_values']
        kv_lens = list(gen_context['kv_lens'])
        ropes = list(gen_context['ropes'])

        curr_kvlen = int(kv_lens[0])
        curr_position_id = int(ropes[0])
        n = len(token_ids)

        packed_text_ids = torch.tensor(list(token_ids), dtype=torch.long)
        packed_text_position_ids = torch.tensor(
            list(range(curr_position_id, curr_position_id + n)), dtype=torch.long
        )
        packed_text_indexes = torch.tensor(
            list(range(curr_kvlen, curr_kvlen + n)), dtype=torch.long
        )
        packed_key_value_indexes = torch.tensor(
            list(range(0, curr_kvlen)), dtype=torch.long
        )
        text_token_lens = torch.tensor([n], dtype=torch.int)
        key_values_lens = torch.tensor([curr_kvlen], dtype=torch.int)

        past_key_values = self.model.forward_cache_update_text(
            past_key_values,
            packed_text_ids=packed_text_ids,
            packed_text_position_ids=packed_text_position_ids,
            text_token_lens=text_token_lens,
            packed_text_indexes=packed_text_indexes,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
        )

        gen_context['kv_lens'] = [curr_kvlen + n]
        gen_context['ropes'] = [curr_position_id + n]
        gen_context['past_key_values'] = past_key_values
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
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
        # print(cfg_renorm_type)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
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

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(
        self,
        gen_context,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
        debug_attn_callback=None,
        return_token_trace: bool = False,
    ):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        text_result = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            debug_attn_callback=debug_attn_callback,
            return_token_trace=return_token_trace,
            **generation_input,
        )
        if return_token_trace:
            unpacked_latent, token_trace = text_result
        else:
            unpacked_latent = text_result
            token_trace = None

        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]

        if return_token_trace:
            for item in token_trace:
                query_ids = item.get("query_token_ids", [])
                predicted_ids = item.get("predicted_token_ids", [])
                item["query_tokens"] = self.tokenizer.convert_ids_to_tokens(query_ids)
                item["predicted_tokens"] = self.tokenizer.convert_ids_to_tokens(predicted_ids)
                item["query_text"] = self.tokenizer.decode(query_ids)
                item["predicted_text"] = self.tokenizer.decode(predicted_ids)
            return output, token_trace
        return output
        
    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        force_image_output=False,

        max_think_token_n=1000,
        max_answer_token_n=512,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=None,
        enable_taylorseer=False,
        max_rounds:int=3,
        debug_attn_callback=None,
        text_debug_attn_callback=None,
        inference_bridge_mask: bool = False,
        force_visual_thinking: bool = False,
        force_no_visual_thinking: bool = False,
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
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

                    if image_shapes is None:
                        image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if image_shapes is None:
                image_shapes = (1024, 1024)

            if understanding_output:
                gen_text = self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n,
                    debug_attn_callback=text_debug_attn_callback,
                )
                output_list.append(gen_text)

            elif force_image_output:
                # Zebra-CoT: skip gen_text, directly generate image from current context
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

            elif force_no_visual_thinking:
                # Force the model to SKIP the bridge-image generation step but
                # keep the structural `<image_start>...<image_end>` wrapper in
                # the KV cache. The wrapper uses the *real* special token IDs
                # `start_of_image` / `end_of_image` (`<|vision_start|>` /
                # `<|vision_end|>`), injected directly into the cache without
                # any chat-turn `<|im_start|>`/`<|im_end|>` wrapping — this is
                # exactly the state SFT'd checkpoints saw in training, except
                # with zero image tokens between the wrapper. No gen_image,
                # no update_context_image, no bridge KV at all.
                #
                # Companion ablation to force_visual_thinking + inference_bridge_mask:
                # isolates the effect of the bridge-generation forward pass itself
                # from the effect of having (or blinding) the bridge in the cache.
                output_list.append("<image_start>")
                gen_context = self._append_raw_token_ids(
                    [self.new_token_ids['start_of_image'],
                     self.new_token_ids['end_of_image']],
                    gen_context,
                )

                answer_text = self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_answer_token_n,
                    debug_attn_callback=text_debug_attn_callback,
                )
                output_list.append(answer_text)

            elif force_visual_thinking:
                # Force the model into the visual-thinking flow without relying
                # on it to emit `<image_start>` itself. Used for vanilla (non-SFT)
                # BAGEL where the model wouldn't otherwise trigger gen_image().
                # The text-side context records `<image_start>...<image_end>` as
                # if the model had emitted them, so the answer decoder sees the
                # standard structural wrapper.
                output_list.append("<image_start>")
                gen_context = self.update_context_text("<image_start>", gen_context)

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
                if inference_bridge_mask:
                    bridge_kv_start = int(gen_context['kv_lens'][0])
                    gen_context = self.update_context_image(img_input, gen_context, vae=not understanding_output)
                    bridge_kv_end = int(gen_context['kv_lens'][0])
                    _zero_bridge_kv(
                        gen_context['past_key_values'],
                        bridge_kv_start,
                        bridge_kv_end,
                        num_image_segments=(2 if not understanding_output else 1),
                    )
                else:
                    gen_context = self.update_context_image(img_input, gen_context, vae=not understanding_output)

                gen_context = self.update_context_text("<image_end>", gen_context)

                answer_text = self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_answer_token_n,
                    debug_attn_callback=text_debug_attn_callback,
                )
                output_list.append(answer_text)

            else:
                rounds = 0
                while rounds < max_rounds:
                    gen_text = self.gen_text(
                        gen_context,
                        do_sample=do_sample,
                        temperature=text_temperature,
                        max_length=max_think_token_n,
                        debug_attn_callback=text_debug_attn_callback,
                    )
                    output_list.append(gen_text)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    
                    if "<image_start>" in gen_text:
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
                        if inference_bridge_mask:
                            bridge_kv_start = int(gen_context['kv_lens'][0])
                            gen_context = self.update_context_image(img_input, gen_context, vae=not understanding_output)
                            bridge_kv_end = int(gen_context['kv_lens'][0])
                            _zero_bridge_kv(
                                gen_context['past_key_values'],
                                bridge_kv_start,
                                bridge_kv_end,
                                num_image_segments=(2 if not understanding_output else 1),
                            )
                        else:
                            gen_context = self.update_context_image(img_input, gen_context, vae=not understanding_output)
                        rounds += 1
                    else:
                        if "<answer>" not in gen_text:
                            answer_text = self.gen_text(
                                gen_context,
                                do_sample=do_sample,
                                temperature=text_temperature,
                                max_length=max_answer_token_n,
                                debug_attn_callback=text_debug_attn_callback,
                            )
                            output_list.append(answer_text)
                            gen_context = self.update_context_text(answer_text, gen_context)
                        break

        return output_list
    
    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None, 
        input_list = None, 
        **kargs
    ) -> Dict[str, Any]:
    
        if input_list is None:
            output_dict = {'image': None, 'text': None}

            if image is None and text is None:
                print('Please provide at least one input: either an image or text.')
                return output_dict

            input_list = []
            if image is not None:
                input_list.append(image)
            if text is not None:
                input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)

        return output_list
