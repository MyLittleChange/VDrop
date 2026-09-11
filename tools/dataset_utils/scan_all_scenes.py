"""
Scan all scenes under BLEND_BASE_PATH and collect valid ones with two cameras.

A scene is valid if it has:
  - cameras.json with both camera_0_0 and camera_1_0
  - visible_objects.json
  - coarse/scene.blend or coarse/scene.blend.zip

Usage:
    python scripts/scan_all_scenes.py
    python scripts/scan_all_scenes.py --output_file /path/to/scanned_scenes.json
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

BLEND_BASE_PATH = Path(
    "/path/to/scratch/infinigen/infinigen_debang/infinigen/v00_filtered"
)


def scan_scene_dir(scene_dir: Path, version_folder: str, room_part: str):
    """Validate a single scene directory. Returns (record, error_reason)."""
    scene_id = scene_dir.name

    # Check cameras.json
    cameras_path = scene_dir / "cameras.json"
    if not cameras_path.exists():
        return None, "cameras.json missing"

    try:
        with open(cameras_path) as f:
            cameras = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return None, f"cameras.json unreadable: {e}"

    if "camera_0_0" not in cameras:
        return None, "cameras.json missing camera_0_0"
    if "camera_1_0" not in cameras:
        return None, "cameras.json missing camera_1_0"

    # Check visible_objects.json
    if not (scene_dir / "visible_objects.json").exists():
        return None, "visible_objects.json missing"

    # Check blend file
    coarse_dir = scene_dir / "coarse"
    has_blend = (coarse_dir / "scene.blend").exists()
    has_zip = (coarse_dir / "scene.blend.zip").exists()
    if not (has_blend or has_zip):
        return None, "no scene.blend or scene.blend.zip in coarse/"

    return {
        "scene_id": scene_id,
        "room_part": room_part,
        "version_folder": version_folder,
        "has_blend_file": has_blend,
        "has_blend_zip": has_zip,
    }, None


def main():
    parser = argparse.ArgumentParser(
        description="Scan all scenes and collect valid ones with two cameras"
    )
    parser.add_argument(
        "--blend_base_path",
        default=str(BLEND_BASE_PATH),
        help="Root directory containing version folders",
    )
    parser.add_argument(
        "--output_file",
        default="/path/to/scratch/VisualCoT/orbit_dataset/scanned_scenes.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    base = Path(args.blend_base_path)
    valid_scenes = []
    invalid_scenes = []
    total_scanned = 0

    # List version folders
    version_folders = sorted(
        [d.name for d in base.iterdir() if d.is_dir()]
    )
    print(f"Found {len(version_folders)} version folders: {version_folders}")

    for vf in version_folders:
        vf_path = base / vf
        # List room_part directories
        room_parts = sorted(
            [d.name for d in vf_path.iterdir() if d.is_dir()]
        )

        for rp in room_parts:
            rp_path = vf_path / rp
            # List scene directories
            try:
                scene_dirs = sorted(
                    [d for d in rp_path.iterdir() if d.is_dir()]
                )
            except PermissionError:
                continue

            for scene_dir in scene_dirs:
                total_scanned += 1
                record, error = scan_scene_dir(scene_dir, vf, rp)
                if record:
                    valid_scenes.append(record)
                else:
                    invalid_scenes.append({
                        "path": str(scene_dir),
                        "reason": error,
                    })

        print(f"  {vf}: scanned, {len(valid_scenes)} valid so far")

    # Check for duplicate scene_ids
    scene_id_counts = {}
    for s in valid_scenes:
        scene_id_counts[s["scene_id"]] = scene_id_counts.get(s["scene_id"], 0) + 1
    duplicates = {k: v for k, v in scene_id_counts.items() if v > 1}

    # Save output
    output = {
        "metadata": {
            "blend_base_path": str(base),
            "scan_date": datetime.now().isoformat(),
            "total_dirs_scanned": total_scanned,
            "valid_scenes": len(valid_scenes),
            "invalid_scenes": len(invalid_scenes),
            "unique_scene_ids": len(scene_id_counts),
            "duplicate_scene_ids": len(duplicates),
        },
        "scenes": valid_scenes,
        "invalid_scenes": invalid_scenes[:100],  # cap to avoid huge output
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nScan complete:")
    print(f"  Total directories scanned: {total_scanned}")
    print(f"  Valid scenes: {len(valid_scenes)}")
    print(f"  Invalid scenes: {len(invalid_scenes)}")
    print(f"  Unique scene_ids: {len(scene_id_counts)}")
    print(f"  Duplicate scene_ids: {len(duplicates)}")
    if duplicates:
        print(f"  Top duplicates: {list(duplicates.items())[:10]}")
    print(f"  Saved to: {args.output_file}")


if __name__ == "__main__":
    main()
