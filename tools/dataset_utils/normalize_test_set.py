"""
Normalize spatial dataset to always reference "second image" perspective,
remap image paths, and resolve topdown map paths.

Usage:
    python scripts/normalize_test_set.py
    python scripts/normalize_test_set.py --input_file /path/to/dataset.json --output_file /path/to/output.json
"""

import argparse
import json
import os


TOPDOWN_DIR = "/path/to/scratch/VisualCoT/infinigen/topdown_maps_rel/topdown_approved_mcqs_relative_distance"


def remap_path(path: str) -> str:
    """Remap paths from home directories to network scratch."""
    if path is None:
        return None
    path = path.replace(
        "/path/to/scratch",
        "/path/to/scratch",
    )
    path = path.replace(
        "/path/to/scratch",
        "/path/to/scratch",
    )
    return path


def main():
    parser = argparse.ArgumentParser(description="Normalize test set to second image perspective")
    parser.add_argument(
        "--input_file",
        default="/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance.json",
    )
    parser.add_argument(
        "--output_file",
        default="/path/to/scratch/VisualCoT/spatial_collab_dataset/approved_mcqs_relative_distance_normalized.json",
    )
    parser.add_argument(
        "--topdown_dir",
        default=TOPDOWN_DIR,
    )
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {args.input_file}")

    swapped_count = 0
    already_second_count = 0
    missing_topdown = 0

    for sample in data:
        q = sample["question_both_views"]
        has_first_image = "first image" in q.lower()
        # user_1_question set + user_2_question null = question is from user_1's perspective
        is_user1_perspective = (
            sample.get("user_1_question") is not None
            and sample.get("user_2_question") is None
        )
        needs_swap = has_first_image or is_user1_perspective

        if needs_swap:
            # Update question text
            # if has_first_image:
            #     sample["question_both_views"] = q.replace("first image", "second image").replace("First image", "Second image")
            # else:
            #     # "Relative to you" style — prepend second image reference
            #     sample["question_both_views"] = "From the perspective of the second image, " + q[0].lower() + q[1:]

            # Swap image paths
            sample["user_1_image_local_path"], sample["user_2_image_local_path"] = (
                sample["user_2_image_local_path"],
                sample["user_1_image_local_path"],
            )
            # Swap question fields
            sample["user_1_question"], sample["user_2_question"] = (
                sample["user_2_question"],
                sample["user_1_question"],
            )
            # Move answer data from user_1 to user_2
            sample["options_user_2"] = sample["options_user_1"]
            sample["user_2_gt_answer_idx"] = sample["user_1_gt_answer_idx"]
            sample["user_2_gt_answer_text"] = sample["user_1_gt_answer_text"]
            # Clear user_1 fields
            sample["options_user_1"] = None
            sample["user_1_gt_answer_idx"] = None
            sample["user_1_gt_answer_text"] = None
            topdown_agent = 1
            swapped_count += 1
        else:
            topdown_agent = 2
            already_second_count += 1

        # Ensure all questions mention "second image"
        # if "second image" not in sample["question_both_views"].lower():
        #     sample["question_both_views"] = (
        #         "From the perspective of the second image, "
        #         + sample["question_both_views"][0].lower()
        #         + sample["question_both_views"][1:]
        #     )

        # Remap image paths
        sample["user_1_image_local_path"] = remap_path(sample["user_1_image_local_path"])
        sample["user_2_image_local_path"] = remap_path(sample["user_2_image_local_path"])

        # Resolve topdown path
        scene_id = sample["scene_id"]
        sample_id = sample["sample_id"]
        topdown_path = os.path.join(
            args.topdown_dir, scene_id, f"topdown_agent_{topdown_agent}_{sample_id}.png"
        )
        if os.path.exists(topdown_path):
            sample["topdown_path"] = topdown_path
        else:
            sample["topdown_path"] = None
            missing_topdown += 1

    # Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=2)

    # Verify
    has_second = sum(1 for s in data if "second image" in s["question_both_views"].lower())
    has_first = sum(1 for s in data if "first image" in s["question_both_views"].lower())
    has_options_2 = sum(1 for s in data if s.get("options_user_2") is not None)
    has_options_1_only = sum(1 for s in data if s.get("options_user_1") is not None and s.get("options_user_2") is None)
    has_topdown = sum(1 for s in data if s.get("topdown_path") is not None)

    print(f"\nTotal samples: {len(data)}")
    print(f"Swapped: {swapped_count}")
    print(f"Already second image: {already_second_count}")
    print(f"Missing topdown maps: {missing_topdown}")
    print(f"\nVerification:")
    print(f"  Mention 'second image': {has_second}")
    print(f"  Mention 'first image': {has_first}")
    print(f"  Have options_user_2: {has_options_2}")
    print(f"  Have options_user_1 only: {has_options_1_only}")
    print(f"  Have topdown_path: {has_topdown}")
    print(f"\nSaved to: {args.output_file}")


if __name__ == "__main__":
    main()
