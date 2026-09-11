"""
Utility for registering spatial image special tokens on top of the base BAGEL tokens.

Usage:
    from data.data_utils import add_special_tokens
    from SpatialUnderstanding.spatial_tokens import add_spatial_special_tokens

    tokenizer, new_token_ids, num_new = add_special_tokens(tokenizer)
    tokenizer, new_token_ids, num_spatial_new = add_spatial_special_tokens(tokenizer, new_token_ids)
"""


def add_spatial_special_tokens(tokenizer, existing_token_ids=None):
    """
    Register <|spatial_image_start|> and <|spatial_image_end|> as special tokens.

    Args:
        tokenizer: The tokenizer to add tokens to.
        existing_token_ids: Dict of existing token IDs (from add_special_tokens).
                           If provided, the new IDs are merged into this dict.

    Returns:
        tokenizer: Updated tokenizer.
        new_token_ids: Dict with all token IDs (existing + spatial).
        num_new_tokens: Number of newly added tokens.
    """
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    # Also check tokens already in the vocab (added via add_tokens)
    new_tokens = []
    for token in ['<|spatial_image_start|>', '<|spatial_image_end|>']:
        if token not in all_special_tokens and tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id:
            new_tokens.append(token)

    num_new_tokens = tokenizer.add_tokens(new_tokens)

    start_of_spatial_image = tokenizer.convert_tokens_to_ids('<|spatial_image_start|>')
    end_of_spatial_image = tokenizer.convert_tokens_to_ids('<|spatial_image_end|>')

    if existing_token_ids is not None:
        new_token_ids = dict(existing_token_ids)
    else:
        new_token_ids = {}

    new_token_ids['start_of_spatial_image'] = start_of_spatial_image
    new_token_ids['end_of_spatial_image'] = end_of_spatial_image

    return tokenizer, new_token_ids, num_new_tokens
