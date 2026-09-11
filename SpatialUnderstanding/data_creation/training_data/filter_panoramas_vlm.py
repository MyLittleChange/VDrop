#!/usr/bin/env python3
"""
Filter bad panoramas using a VLM (Qwen3-VL-32B via vLLM).

Scans all rendered panorama directories, asks the VLM whether each panorama
shows a proper room interior, and writes a blacklist JSON of bad panoramas.

Usage:
    python filter_panoramas_vlm.py \
        --api_base http://localhost:8766/v1 \
        --output_file /path/to/scratch/infinigen/bad_panoramas.json
"""

import argparse
import base64
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

PANORAMA_DIRS = [
    "/path/to/scratch/infinigen/rendered_panorama_v5",
    "/path/to/scratch/infinigen/rendered_panorama_v4",
    "/path/to/scratch/VisualCoT/infinigen/rendered_panorama_train",
]

PROMPT = (
    "Look at this panoramic image. Does it show the interior of a room "
    "(walls, furniture, floor, ceiling visible)? Or does it mostly show "
    "the outside of a building, sky, ground, or is otherwise not a proper "
    "room interior view?\n\n"
    "Answer with exactly one word: GOOD if it shows a room interior, "
    "BAD if it shows outside/sky/ground or is not a proper room view."
)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def check_panorama(client: OpenAI, model: str, image_path: str) -> dict:
    """Ask the VLM whether a panorama is a valid room interior."""
    try:
        b64 = encode_image(image_path)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                            },
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            max_tokens=16,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip().upper()
        is_bad = "BAD" in answer
        return {"path": image_path, "is_bad": is_bad, "answer": answer}
    except Exception as e:
        print(f"[ERROR] {image_path}: {e}")
        return {"path": image_path, "is_bad": False, "answer": f"ERROR: {e}"}


def main():
    parser = argparse.ArgumentParser(description="Filter bad panoramas using VLM")
    parser.add_argument("--api_base", type=str, default="http://localhost:8766/v1")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: auto-detect from vLLM)")
    parser.add_argument("--output_file", type=str,
                        default="/path/to/scratch/infinigen/bad_panoramas.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--panorama_dirs", nargs="+", default=PANORAMA_DIRS)
    args = parser.parse_args()

    client = OpenAI(base_url=args.api_base, api_key="dummy")

    # Auto-detect model name
    model = args.model
    if model is None:
        models = client.models.list()
        model = models.data[0].id
        print(f"Auto-detected model: {model}")

    # Collect all panorama paths
    all_panos = []
    for pano_dir in args.panorama_dirs:
        pngs = sorted(glob.glob(os.path.join(pano_dir, "*/*.png")))
        print(f"  {pano_dir}: {len(pngs)} panoramas")
        all_panos.extend(pngs)
    print(f"Total panoramas to check: {len(all_panos)}")

    # Check each panorama
    results = []
    bad_paths = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(check_panorama, client, model, p): p
            for p in all_panos
        }
        done = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result["is_bad"]:
                bad_paths.append(result["path"])
                print(f"  BAD: {result['path']}")
            done += 1
            if done % 100 == 0:
                print(f"  Progress: {done}/{len(all_panos)} ({len(bad_paths)} bad so far)")

    # Write results
    output = {
        "bad_panoramas": sorted(bad_paths),
        "total_checked": len(all_panos),
        "total_bad": len(bad_paths),
        "details": sorted(results, key=lambda x: x["path"]),
    }
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone! {len(bad_paths)}/{len(all_panos)} bad panoramas")
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
