"""
Export visual_only parquet training data to a portable format:
  - images/<sample_id>_{cam0,cam1,panorama}.png
  - training_data.json  (list of dicts with relative image paths)

Usage:
  python export_training_data.py \
    --parquet_dir /path/to/scratch/infinigen/training_data_mix_all_rotation/visual_only/ \
    --output_dir  /path/to/scratch/infinigen/training_data_mix_all_rotation/training_data_mix_all_balance/ \
    [--workers 16]
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq


def write_image(path: Path, data: bytes):
    path.write_bytes(data)


def export(parquet_dir: str, output_dir: str, workers: int):
    parquet_dir = Path(parquet_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Read chunk ordering from parquet_info.json
    info_path = parquet_dir / "parquet_info.json"
    with open(info_path) as f:
        parquet_info = json.load(f)

    # Sort chunks by filename so ordering is deterministic
    chunk_paths = sorted(parquet_info.keys(), key=lambda p: Path(p).name)

    samples = []
    global_idx = 0

    print(f"Processing {len(chunk_paths)} chunks from {parquet_dir}")

    for chunk_path in chunk_paths:
        table = pq.read_table(chunk_path)
        n = len(table)

        write_tasks = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}

            for row_idx in range(n):
                sample_id = f"sample_{global_idx:06d}"
                global_idx += 1

                image_list = table["image_list"][row_idx].as_py()
                instruction = table["instruction_list"][row_idx].as_py()[0]
                output_parts = table["output_text_list"][row_idx].as_py()
                output = "".join(output_parts)

                cam0_name = f"{sample_id}_cam0.png"
                cam1_name = f"{sample_id}_cam1.png"
                pano_name = f"{sample_id}_panorama.png"

                cam0_path = images_dir / cam0_name
                cam1_path = images_dir / cam1_name
                pano_path = images_dir / pano_name

                futures[executor.submit(write_image, cam0_path, image_list[0])] = None
                futures[executor.submit(write_image, cam1_path, image_list[1])] = None
                futures[executor.submit(write_image, pano_path, image_list[2])] = None

                samples.append({
                    "sample_id": sample_id,
                    "image_cam0": f"images/{cam0_name}",
                    "image_cam1": f"images/{cam1_name}",
                    "image_panorama": f"images/{pano_name}",
                    "instruction": instruction,
                    "output": output,
                })

            for f in as_completed(futures):
                exc = f.exception()
                if exc:
                    print(f"[ERROR] Image write failed: {exc}")

        print(f"  {Path(chunk_path).name}: {n} rows (total so far: {global_idx})")

    out_json = output_dir / "training_data.json"
    with open(out_json, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"\nDone! {len(samples)} samples")
    print(f"  Images: {images_dir}")
    print(f"  JSON:   {out_json}")
    print(f"  Expected image files: {len(samples) * 3}")
    actual = sum(1 for _ in images_dir.iterdir())
    print(f"  Actual  image files: {actual}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dir", required=True,
                        help="Directory containing chunk_*.parquet and parquet_info.json")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory (images/ and training_data.json written here)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Thread workers for parallel image writes")
    args = parser.parse_args()
    export(args.parquet_dir, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
