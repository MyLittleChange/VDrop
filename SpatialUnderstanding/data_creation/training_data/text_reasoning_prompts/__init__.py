"""Per-category prompt modules used by annotate_text_reasoning.py.

The annotator dispatches on the category prefix of a sample id (e.g. ``spatial`` from
``spatial_000123``) and only loads the prompt module for that category. This keeps
each LLM call short — only the in-context examples for the current category travel
in the prompt.
"""

from . import (
    anchor,
    counting,
    perspective_taking,
    point_matching,
    relative_distance,
    spatial,
)

# id-prefix → module
DISPATCH = {
    "anchor": anchor,
    "counting": counting,
    "perspective_taking": perspective_taking,
    "point_matching": point_matching,
    "relative_distance": relative_distance,
    "spatial": spatial,
}
