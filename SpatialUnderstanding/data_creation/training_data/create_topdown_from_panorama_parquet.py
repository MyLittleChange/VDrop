#!/usr/bin/env python3
"""
Create the topdown visual_only parquet by re-keying the existing non-rotation panorama
parquet (training_data_mix_all_balance/visual_only/) with topdown maps swapped into
image_list[2].

This guarantees row-for-row alignment with the panorama set: same instructions, same
outputs, same cam0/cam1 bytes — the only difference is the bridge image content.

Join chain:
  panorama parquet row.image_list[0] (PNG bytes)
    --md5--> no_thinking.jsonl line.image[0] (file on disk)
    --id--> sample_id (e.g. 'v5_relative_distance_003068')
    --strip prefix--> 'relative_distance_003068'
    --topdown_agent_1_{sample_id}.png--> topdown PNG
        (with scene-level fallback: any other PNG in the same scene_id directory,
         since topdowns within one scene share identical content)
"""

import argparse
import hashlib
import io
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


PANORAMA_PARQUET_INFO = (
    "/path/to/scratch/infinigen/training_data_mix_all_balance/"
    "visual_only/parquet_info.json"
)
PANORAMA_PARQUET_PATH_REMAP = (
    "/path/to/scratch/VisualCoT/training_data",
    "/path/to/scratch/infinigen",
)
NO_THINKING_JSONL = (
    "/path/to/scratch/infinigen/training_data_mix_all_balance/"
    "no_thinking/no_thinking.jsonl"
)
JSONL_IMAGE_PATH_PREFIX = "/network/scratch/"
TOPDOWN_DIRS = [
    "/path/to/scratch/infinigen/topdown_maps_v4",
    "/path/to/scratch/infinigen/topdown_maps_v5",
    "/path/to/scratch/VisualCoT/infinigen/topdown_maps_train",
]
DEFAULT_OUTPUT_DIR = (
    "/path/to/scratch/infinigen/"
    "training_data_mix_topdown_no_rotation/visual_only"
)

SOURCE_PREFIXES = ("v4_", "v5_", "ankur_")
SCENE_ID_RE = re.compile(r"^[0-9a-f]{7,9}$")


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def strip_source_prefix(sample_id):
    for p in SOURCE_PREFIXES:
        if sample_id.startswith(p):
            return sample_id[len(p):]
    return sample_id


def extract_scene_id(image_path):
    for part in image_path.split("/"):
        if SCENE_ID_RE.match(part):
            return part
    return None


def remap_panorama_chunk_path(p):
    src, dst = PANORAMA_PARQUET_PATH_REMAP
    return p.replace(src, dst)


def build_jsonl_index(jsonl_path, max_workers):
    """Return {cam0_md5: (sample_id, scene_id)} + stats."""
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "id": d["id"],
                "image0_path": JSONL_IMAGE_PATH_PREFIX + d["image"][0],
            })
    print(f"Loaded {len(rows)} JSONL rows; hashing cam0 files…")

    def hash_one(row):
        try:
            if not os.path.exists(row["image0_path"]):
                return row, None, "no_file"
            md5 = md5_file(row["image0_path"])
            scene = extract_scene_id(row["image0_path"])
            return row, (md5, scene), "ok"
        except Exception as e:
            return row, None, f"error:{e}"

    md5_to_meta = {}
    counts = Counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for row, result, status in tqdm(
            ex.map(hash_one, rows), total=len(rows), desc="hashing JSONL cam0"
        ):
            counts[status] += 1
            if status == "ok":
                md5, scene = result
                if md5 in md5_to_meta:
                    counts["md5_collision"] += 1
                md5_to_meta[md5] = (row["id"], scene)
    print(f"  JSONL hash status: {dict(counts)}")
    print(f"  unique cam0 md5s indexed: {len(md5_to_meta)}")
    return md5_to_meta


def build_topdown_indexes(topdown_dirs):
    """Return (per_sample_index, per_scene_index)."""
    per_sample = {}
    per_scene = {}
    for root in topdown_dirs:
        if not os.path.isdir(root):
            print(f"  warning: topdown dir missing: {root}")
            continue
        for scene in os.listdir(root):
            scene_dir = os.path.join(root, scene)
            if not os.path.isdir(scene_dir):
                continue
            for fname in sorted(os.listdir(scene_dir)):
                if not (fname.startswith("topdown_agent_1_") and fname.endswith(".png")):
                    continue
                sid = fname[len("topdown_agent_1_"):-len(".png")]
                full = os.path.join(scene_dir, fname)
                per_sample[sid] = full
                per_scene.setdefault(scene, full)
    print(f"  topdown index: per_sample={len(per_sample)}, per_scene={len(per_scene)}")
    return per_sample, per_scene


def resolve_topdown(sample_id, scene_id, td_per_sample, td_per_scene):
    sid = strip_source_prefix(sample_id)
    if sid in td_per_sample:
        return td_per_sample[sid], "per_sample"
    if scene_id and scene_id in td_per_scene:
        return td_per_scene[scene_id], "per_scene"
    return None, "missing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panorama_parquet_info", default=PANORAMA_PARQUET_INFO)
    ap.add_argument("--no_thinking_jsonl", default=NO_THINKING_JSONL)
    ap.add_argument("--topdown_dirs", nargs="+", default=TOPDOWN_DIRS)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--max_workers", type=int, default=16)
    ap.add_argument("--rows_per_group", type=int, default=100)
    ap.add_argument("--groups_per_file", type=int, default=10)
    args = ap.parse_args()

    print(f"Building JSONL cam0-md5 index from {args.no_thinking_jsonl}")
    md5_to_meta = build_jsonl_index(args.no_thinking_jsonl, args.max_workers)

    print("\nBuilding topdown indexes")
    td_per_sample, td_per_scene = build_topdown_indexes(args.topdown_dirs)

    print(f"\nLoading panorama parquet info: {args.panorama_parquet_info}")
    info = json.load(open(args.panorama_parquet_info))
    chunk_paths = sorted(info.keys(), key=lambda k: int(
        os.path.basename(k).split("chunk_")[1].split(".")[0]
    ))
    chunk_paths = [remap_panorama_chunk_path(p) for p in chunk_paths]
    print(f"  {len(chunk_paths)} panorama chunks")

    counts = Counter()
    new_rows = []

    for cp in tqdm(chunk_paths, desc="processing chunks"):
        if not os.path.exists(cp):
            counts["chunk_missing"] += 1
            continue
        t = pq.read_table(cp)
        rows = t.to_pylist()
        for row in rows:
            counts["total_rows"] += 1
            cam0_bytes = row["image_list"][0]
            cam0_md5 = md5_bytes(cam0_bytes)
            meta = md5_to_meta.get(cam0_md5)
            if meta is None:
                counts["no_md5_match"] += 1
                continue
            sample_id, scene_id = meta
            td_path, td_source = resolve_topdown(
                sample_id, scene_id, td_per_sample, td_per_scene
            )
            if td_path is None:
                counts[f"no_topdown_{td_source}"] += 1
                continue
            counts[f"ok_{td_source}"] += 1
            try:
                with open(td_path, "rb") as f:
                    td_bytes = f.read()
            except Exception as e:
                counts["topdown_read_error"] += 1
                continue
            new_rows.append({
                "image_list": [cam0_bytes, row["image_list"][1], td_bytes],
                "instruction_list": row["instruction_list"],
                "output_text_list": row["output_text_list"],
            })

    print(f"\n=== summary ===")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")
    print(f"  rows to write: {len(new_rows)}")

    if not new_rows:
        print("nothing to write, exiting")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    schema = pa.schema([
        pa.field("image_list", pa.list_(pa.binary())),
        pa.field("instruction_list", pa.list_(pa.string())),
        pa.field("output_text_list", pa.list_(pa.string())),
    ])

    rows_per_file = args.rows_per_group * args.groups_per_file
    parquet_info = {}
    for chunk_idx, start in enumerate(tqdm(
        range(0, len(new_rows), rows_per_file), desc="writing chunks"
    )):
        chunk_rows = new_rows[start:start + rows_per_file]
        out_path = os.path.join(args.output_dir, f"chunk_{chunk_idx}.parquet")
        num_groups = 0
        with pq.ParquetWriter(out_path, schema=schema, version="2.6") as w:
            for j in range(0, len(chunk_rows), args.rows_per_group):
                group = chunk_rows[j:j + args.rows_per_group]
                w.write_table(pa.Table.from_pylist(group, schema=schema))
                num_groups += 1
        parquet_info[out_path] = {
            "num_row_groups": num_groups,
            "num_rows": len(chunk_rows),
        }

    info_path = os.path.join(args.output_dir, "parquet_info.json")
    with open(info_path, "w") as f:
        json.dump(parquet_info, f, indent=2)

    print(f"\nDone! {len(new_rows)} rows in {len(parquet_info)} chunk(s) at {args.output_dir}")
    print(f"parquet_info: {info_path}")


if __name__ == "__main__":
    main()
