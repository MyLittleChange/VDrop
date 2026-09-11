"""
Utility for registering dynamic visual mode tokens on top of the base BAGEL tokens.

The mode token is a plain text token the model generates *before* <image_start>.
It signals which visual representation to render and at what resolution:

    <panoramic>   → 720×1024  wide-angle panoramic view
                    Used for: anchor, counting, spatial_orientation
    <BEV>         → 720×720   bird's-eye-view / top-down distance map
                    Used for: closest, farthest
    <zoom-in-out> → 720×2048  zoomed perspective (reserved for future use)

Usage:
    from data.data_utils import add_special_tokens
    from SpatialUnderstanding.dynamic_visual_tokens import add_dynamic_visual_tokens

    tokenizer, new_token_ids, num_new = add_special_tokens(tokenizer)
    tokenizer, new_token_ids, num_dyn_new = add_dynamic_visual_tokens(tokenizer, new_token_ids)
"""

DYNAMIC_VISUAL_MODES = {
    '<panoramic>':   (720, 1024),
    '<BEV>':         (720, 720),
    '<zoom-in-out>': (720, 2048),
}

_TOKEN_KEY_MAP = {
    '<panoramic>':   'panoramic_token_id',
    '<BEV>':         'bev_token_id',
    '<zoom-in-out>': 'zoom_in_out_token_id',
}


def add_dynamic_visual_tokens(tokenizer, existing_token_ids=None):
    """
    Register <panoramic>, <BEV>, and <zoom-in-out> as special tokens.

    Args:
        tokenizer: The tokenizer to add tokens to.
        existing_token_ids: Dict of existing token IDs (from add_special_tokens).
                           If provided, the new IDs are merged into this dict.

    Returns:
        tokenizer: Updated tokenizer.
        new_token_ids: Dict with all token IDs (existing + dynamic mode tokens).
                       Also contains 'dynamic_mode_to_size': {token_id: (H, W)}.
        num_new_tokens: Number of newly added tokens.
    """
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []
    for token in DYNAMIC_VISUAL_MODES:
        if token not in all_special_tokens and tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id:
            new_tokens.append(token)

    num_new_tokens = tokenizer.add_tokens(new_tokens)

    if existing_token_ids is not None:
        new_token_ids = dict(existing_token_ids)
    else:
        new_token_ids = {}

    dynamic_mode_to_size = {}
    for token, key in _TOKEN_KEY_MAP.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        new_token_ids[key] = token_id
        dynamic_mode_to_size[token_id] = DYNAMIC_VISUAL_MODES[token]

    new_token_ids['dynamic_mode_to_size'] = dynamic_mode_to_size

    return tokenizer, new_token_ids, num_new_tokens
