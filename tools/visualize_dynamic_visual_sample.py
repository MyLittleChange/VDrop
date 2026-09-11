"""
Visualize a random sample from the dynamic_visual_only dataset.

- Panoramic samples: input perspective views + generated panoramic image (wide, ~2048px wide)
- BEV samples: input perspective views + generated BEV image (square, ~720x720)
"""

import glob
import io
import random
import textwrap

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from PIL import Image


DATA_DIR = "/path/to/scratch/VisualCoT/training_data/mix_view_qa_Cosmix_only/dynamic_visual_only"


def load_random_sample(view_type: str | None = None):
    """Load a random sample. view_type: 'BEV', 'panoramic', or None for either."""
    chunks = sorted(glob.glob(f"{DATA_DIR}/chunk_*.parquet"))
    random.shuffle(chunks)

    for chunk_path in chunks:
        df = pd.read_parquet(chunk_path)
        indices = list(range(len(df)))
        random.shuffle(indices)
        for i in indices:
            row = df.iloc[i]
            output = "".join(row["output_text_list"])
            if view_type is None:
                return row
            if view_type == "BEV" and "<BEV>" in output:
                return row
            if view_type == "panoramic" and "<panoramic>" in output:
                return row

    raise ValueError(f"No sample found for view_type={view_type!r}")


def bytes_to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b))


def get_view_type(row) -> str:
    output = "".join(row["output_text_list"])
    if "<BEV>" in output:
        return "BEV"
    if "<panoramic>" in output:
        return "panoramic"
    return "unknown"


def get_answer(row) -> str:
    output = "".join(row["output_text_list"])
    import re
    m = re.search(r"<answer>(.*?)</answer>", output)
    return m.group(1) if m else "?"


def visualize(row):
    images = [bytes_to_pil(b) for b in row["image_list"]]
    vtype = get_view_type(row)
    answer = get_answer(row)
    question = row["instruction_list"][0]

    # The generated view is always the last image
    # Input perspective views are all images before the last
    input_images = images[:-1]
    generated_image = images[-1]

    n_input = len(input_images)
    n_cols = max(n_input, 1)

    fig = plt.figure(figsize=(5 * n_cols, 10))
    gs = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.35, wspace=0.1)

    # Row 0: input perspective views
    for j, img in enumerate(input_images):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img)
        ax.set_title(f"Input view {j+1}", fontsize=10)
        ax.axis("off")

    # Row 1: generated view (spans all columns)
    ax_gen = fig.add_subplot(gs[1, :])
    ax_gen.imshow(generated_image)
    label = "Generated BEV (top-down)" if vtype == "BEV" else "Generated Panoramic (wide-angle)"
    ax_gen.set_title(label, fontsize=11, fontweight="bold")
    ax_gen.axis("off")

    # Question + answer as suptitle
    q_short = "\n".join(textwrap.wrap(question.split("\n", 1)[-1], width=100))
    fig.suptitle(
        f"[{vtype}]  Answer: {answer}\n\n{q_short}",
        fontsize=9,
        y=1.02,
        ha="left",
        x=0.01,
        wrap=True,
    )

    plt.savefig("sample_visualization.png", bbox_inches="tight", dpi=100)
    print(f"Saved to sample_visualization.png")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        choices=["BEV", "panoramic"],
        default='BEV',
        help="Filter by view type (default: random)",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.type is None:
        choice = input("Choose view type [BEV / panoramic / random]: ").strip().lower()
        if choice in ("bev", "b"):
            args.type = "BEV"
        elif choice in ("panoramic", "pano", "p"):
            args.type = "panoramic"
        # else: None → random

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Loading random {'any' if args.type is None else args.type} sample...")
    row = load_random_sample(view_type=args.type)
    vtype = get_view_type(row)
    print(f"View type: {vtype}")
    print(f"Num images: {len(row['image_list'])}")
    print(f"Question: {row['instruction_list'][0][:200]}...")
    print(f"Output: {row['output_text_list']}")
    visualize(row)
