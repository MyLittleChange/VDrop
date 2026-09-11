"""
Training entry point that extends the base training setup with spatial image tokens.

This script wraps the standard training flow from train/pretrain_unified_navit.py,
adding:
  1. Registration of <|spatial_image_start|> / <|spatial_image_end|> tokens
  2. Embedding resize to accommodate new tokens
  3. Use of SpatialPackedDataset instead of PackedDataset

Usage:
    # In your training shell script, replace:
    #   python train/pretrain_unified_navit.py ...
    # with:
    #   python SpatialUnderstanding/spatial_train.py ...
    # (all arguments remain the same)

Integration guide:
    If you prefer to integrate into pretrain_unified_navit.py directly, make these
    3 changes after the `add_special_tokens(tokenizer)` call:

    1. Add tokens:
        from SpatialUnderstanding.spatial_tokens import add_spatial_special_tokens
        tokenizer, new_token_ids, num_spatial_new = add_spatial_special_tokens(tokenizer, new_token_ids)
        if num_spatial_new > 0:
            model.language_model.resize_token_embeddings(len(tokenizer))
            model.config.llm_config.vocab_size = len(tokenizer)
            model.language_model.config.vocab_size = len(tokenizer)

    2. Use SpatialPackedDataset:
        from SpatialUnderstanding.spatial_dataset import SpatialPackedDataset
        train_dataset = SpatialPackedDataset(...)  # instead of PackedDataset(...)

    3. (Optional) Use SpatialBagel for inference:
        from SpatialUnderstanding.spatial_bagel import wrap_as_spatial_bagel
        wrap_as_spatial_bagel(model)
        model.set_spatial_tokens(new_token_ids['start_of_spatial_image'],
                                 new_token_ids['end_of_spatial_image'])
"""

import sys
import os

# Ensure project root is on the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def patch_training_setup():
    """
    Monkey-patch the training module to use spatial tokens and dataset.

    This patches:
    - add_special_tokens -> also adds spatial tokens
    - PackedDataset -> SpatialPackedDataset
    """
    import train.pretrain_unified_navit as train_module
    from data.data_utils import add_special_tokens as original_add_special_tokens
    from SpatialUnderstanding.spatial_tokens import add_spatial_special_tokens
    from SpatialUnderstanding.spatial_dataset import SpatialPackedDataset

    # Wrap add_special_tokens to also add spatial tokens
    def add_special_tokens_with_spatial(tokenizer):
        tokenizer, new_token_ids, num_new = original_add_special_tokens(tokenizer)
        tokenizer, new_token_ids, num_spatial_new = add_spatial_special_tokens(tokenizer, new_token_ids)
        return tokenizer, new_token_ids, num_new + num_spatial_new

    # Patch the training module
    train_module.add_special_tokens = add_special_tokens_with_spatial
    train_module.PackedDataset = SpatialPackedDataset


if __name__ == "__main__":
    patch_training_setup()

    # Import and run the original training main function
    from train.pretrain_unified_navit import main
    main()
