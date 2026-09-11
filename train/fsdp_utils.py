# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import os

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file

from modeling.bagel.modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.bagel.qwen2_navit import (
    Qwen2DecoderLayer, 
    Qwen2MoEDecoderLayer, 
    Qwen2MoTDecoderLayer,
)
from modeling.bagel.siglip_navit import SiglipEncoderLayer, SiglipVisionTransformer


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy,
        backward_prefetch,
        cpu_offload,
        num_replicate,
        num_shard=8,
        use_lora=False,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard
        self.use_lora = use_lora


def fsdp_wrapper(original_model, fsdp_config, ignored_modules=[]):
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    else:
        device_mesh = None
    # LoRA mixes frozen (base) and trainable (lora_A/lora_B) params inside the
    # same decoder layer, so FSDP's default FlatParameter (use_orig_params=False)
    # rejects wrapping with "Must flatten tensors with uniform requires_grad".
    # use_orig_params=True keeps params as distinct tensors and is required here.
    use_orig_params = getattr(fsdp_config, "use_lora", False)
    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Qwen2DecoderLayer,
                Qwen2MoEDecoderLayer,
                Qwen2MoTDecoderLayer,
                SiglipEncoderLayer,
                SiglipVisionTransformer,
                MLPconnector,
                TimestepEmbedder,
                PositionEmbedding,
            },
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
        use_orig_params=use_orig_params,
    )


class FSDPCheckpoint:
    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir, 
        train_steps, 
        model, 
        ema_model, 
        optimizer, 
        scheduler, 
        data_status,
        logger, 
        fsdp_config,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        # Reclaim the CUDA caching allocator before FSDP's state_dict all-gather.
        # Each `_all_gather_flat_param` allocates an unsharded flat param on every
        # rank; without this, leftover activations / comm buffers from the prior
        # train step can leave the allocator too fragmented to find a contiguous
        # block, surfacing as "ncclUnhandledCudaError: Cuda failure 2 'out of memory'".
        gc.collect()
        torch.cuda.empty_cache()

        if ema_model is not None:
            with FSDP.state_dict_type(
                ema_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                ema_state_dict = ema_model.state_dict()
                if dist.get_rank() == 0:
                    # Convert to bf16 before saving to reduce checkpoint size
                    ema_state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in ema_state_dict.items()}
                    save_file(ema_state_dict_bf16, os.path.join(save_path, "ema.safetensors"))
                    del ema_state_dict_bf16
                del ema_state_dict
            gc.collect()
            torch.cuda.empty_cache()

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            model_state_dict = model.state_dict()
            if dist.get_rank() == 0:
                if getattr(fsdp_config, "use_lora", False):
                    # LoRA: save only adapter weights + the unfrozen embed_tokens / lm_head.
                    lora_state_dict = {
                        k: v.to(torch.bfloat16)
                        for k, v in model_state_dict.items()
                        if ("lora_" in k)
                        or k.endswith("language_model.model.embed_tokens.weight")
                        or k.endswith("language_model.lm_head.weight")
                    }
                    save_file(lora_state_dict, os.path.join(save_path, "lora_adapter.safetensors"))
                    logger.info(f"Saved LoRA adapter with {len(lora_state_dict)} tensors.")
                    del lora_state_dict
                else:
                    # Convert to bf16 before saving to reduce checkpoint size
                    model_state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in model_state_dict.items()}
                    save_file(model_state_dict_bf16, os.path.join(save_path, "model.safetensors"))
                    del model_state_dict_bf16
            del model_state_dict
        gc.collect()
        torch.cuda.empty_cache()

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_save_path = os.path.join(
                save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                torch.save(optimizer.state_dict(), optimizer_save_path)
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                if dist.get_rank() < fsdp_config.num_shard:
                    torch.save(optimizer.state_dict(), optimizer_save_path)
            else:
                raise NotImplementedError

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if dist.get_rank() == 0 and data_status is not None:
            torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        dist.barrier()
        return

    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            if resume_from_ema:
                model_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
                if not os.path.exists(model_state_dict_path):
                    logger.info(f"ema.safetensors not found, falling back to model.safetensors")
                    model_state_dict_path = os.path.join(resume_from, f"model.safetensors")
            else:
                model_state_dict_path = os.path.join(resume_from, f"model.safetensors")
            model_state_dict = load_file(model_state_dict_path, device="cpu")
            # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
            # which makes it easier to adapt to different resolutions.
            model_state_dict.pop('latent_pos_embed.pos_embed')
            model_state_dict.pop('vit_pos_embed.pos_embed')
            ckpt_vocab_size = model_state_dict['language_model.model.embed_tokens.weight'].shape[0]
            if ckpt_vocab_size != model.language_model.config.vocab_size:
                model.language_model.resize_token_embeddings(ckpt_vocab_size)
                model.config.llm_config.vocab_size = ckpt_vocab_size
                model.language_model.config.vocab_size = ckpt_vocab_size
            msg = model.load_state_dict(model_state_dict, strict=False)
            logger.info(msg)
            del model_state_dict

            if ema_model is not None:
                ema_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
                if not os.path.exists(ema_state_dict_path):
                    logger.info(f"replicaing ema model from {model_state_dict_path}.")
                    ema_state_dict_path = model_state_dict_path
                ema_state_dict = load_file(ema_state_dict_path, device="cpu")
                # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
                # which makes it easier to adapt to different resolutions.
                ema_state_dict.pop('latent_pos_embed.pos_embed')
                ema_state_dict.pop('vit_pos_embed.pos_embed')
                ckpt_vocab_size = ema_state_dict['language_model.model.embed_tokens.weight'].shape[0]
                if ckpt_vocab_size != ema_model.language_model.config.vocab_size:
                    ema_model.language_model.resize_token_embeddings(ckpt_vocab_size)
                    ema_model.config.llm_config.vocab_size = ckpt_vocab_size
                    ema_model.language_model.config.vocab_size = ckpt_vocab_size
                msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                logger.info(msg)
                del ema_state_dict
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_lora_adapter(resume_from, logger, model):
        """Load LoRA adapter weights into a PEFT-wrapped model on CPU.

        Call this AFTER `get_peft_model` but BEFORE FSDP wrap. If `resume_from`
        has no `lora_adapter.safetensors`, this is a no-op (fresh LoRA init).
        """
        if resume_from is None or not os.path.exists(resume_from):
            return model
        adapter_path = os.path.join(resume_from, "lora_adapter.safetensors")
        if not os.path.exists(adapter_path):
            return model
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        adapter_state_dict = load_file(adapter_path, device="cpu")
        # language_model was wrapped with get_peft_model(), so adapter keys live
        # under "language_model.<peft path>". Just load the whole thing onto the
        # top-level Bagel model with strict=False.
        msg = model.load_state_dict(adapter_state_dict, strict=False)
        logger.info(f"LoRA adapter load: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
        del adapter_state_dict
        return model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
            scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = os.path.join(resume_from, "data_status.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status


def grad_checkpoint_check_fn(module):
    module_options = (
        Qwen2DecoderLayer, 
        SiglipEncoderLayer, 
        MLPconnector, 
        Qwen2MoEDecoderLayer, 
        Qwen2MoTDecoderLayer
    )
    return isinstance(module, module_options)


def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=[]):
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules)
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)
