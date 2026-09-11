#!/usr/bin/env python3
"""
Create a YAML configuration file for LLaMA-Factory training.
"""

import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(
        description='Create LLaMA-Factory training YAML config'
    )
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the dataset in dataset_info.json')
    parser.add_argument('--model_path', type=str,
                        default='Qwen/Qwen3-VL-4B-Instruct',
                        help='Path to the base model')
    parser.add_argument('--template', type=str, default='qwen3_vl_nothink',
                        help='Template name for the model')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints')
    parser.add_argument('--output_yaml', type=str, required=True,
                        help='Output YAML file path')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Per-device train batch size')
    parser.add_argument('--grad_accum', type=int, default=8,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=float, default=3.0,
                        help='Number of training epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--cutoff_len', type=int, default=2048,
                        help='Maximum sequence length')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every N steps')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--report_to', type=str, default='none',
                        choices=['none', 'wandb', 'tensorboard'],
                        help='Where to report training metrics')

    args = parser.parse_args()

    config = {
        # Model
        'model_name_or_path': args.model_path,
        'image_max_pixels': 262144,
        'video_max_pixels': 16384,
        'trust_remote_code': True,

        # Method
        'stage': 'sft',
        'do_train': True,
        'finetuning_type': 'lora',
        'lora_rank': args.lora_rank,
        'lora_target': 'all',

        # Dataset
        'dataset': args.dataset_name,
        'template': args.template,
        'cutoff_len': args.cutoff_len,
        'overwrite_cache': True,
        'preprocessing_num_workers': 16,
        'dataloader_num_workers': 4,

        # Output
        'output_dir': args.output_dir,
        'logging_steps': args.logging_steps,
        'save_steps': args.save_steps,
        'plot_loss': True,
        'overwrite_output_dir': True,
        'save_only_model': False,
        'report_to': args.report_to,

        # Train
        'per_device_train_batch_size': args.batch_size,
        'gradient_accumulation_steps': args.grad_accum,
        'learning_rate': args.learning_rate,
        'num_train_epochs': args.num_epochs,
        'lr_scheduler_type': 'cosine',
        'warmup_ratio': args.warmup_ratio,
        'bf16': True,
        'ddp_timeout': 180000000,
        'resume_from_checkpoint': None,
    }

    if args.resume_from_checkpoint:
        config['resume_from_checkpoint'] = args.resume_from_checkpoint

    # Write YAML file
    with open(args.output_yaml, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created training config: {args.output_yaml}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model_path}")
    print(f"Template: {args.template}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size} x {args.grad_accum} (effective: {args.batch_size * args.grad_accum})")


if __name__ == '__main__':
    main()
