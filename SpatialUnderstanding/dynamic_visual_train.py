"""
Training entry point that adds dynamic visual mode tokens to the base training setup.

The mode token (<panoramic>, <BEV>, <zoom-in-out>) is a plain text token that the
model generates *before* <image_start>. Training data format:

    output_text_list: ["<BEV> <image_start>", "<image_end><answer>A</answer>"]

The base PackedDataset handles this normally — <image_start> triggers the vae_image
pack branch as usual, and the mode token is packed as a regular text token just before
it. No custom dataset class is needed.

Usage:
    Replace `train/pretrain_unified_navit.py` with this script in your training command:

        torchrun ... SpatialUnderstanding/dynamic_visual_train.py [same args]
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def patch_training_setup():
    """Patch add_special_tokens to also register dynamic visual mode tokens."""
    import train.pretrain_unified_navit as train_module
    from data.data_utils import add_special_tokens as original_add_special_tokens
    from SpatialUnderstanding.dynamic_visual_tokens import add_dynamic_visual_tokens

    def add_special_tokens_with_dynamic(tokenizer):
        tokenizer, new_token_ids, num_new = original_add_special_tokens(tokenizer)
        tokenizer, new_token_ids, num_dyn_new = add_dynamic_visual_tokens(tokenizer, new_token_ids)
        return tokenizer, new_token_ids, num_new + num_dyn_new

    train_module.add_special_tokens = add_special_tokens_with_dynamic


if __name__ == "__main__":
    patch_training_setup()

    from train.pretrain_unified_navit import main
    main()
