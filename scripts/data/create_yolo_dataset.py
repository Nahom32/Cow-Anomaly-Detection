from __future__ import annotations

import os
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scripts.config import CONFIG
from scripts.data.download_dataset import download_dataset
from scripts.data.splits import SPLIT_NAMES, resolve_split, split_params
from scripts.utils.seeding import set_seed


def create_yolo_dataset(
    output_dir: str = "/content/cow_detection_dataset_fixed",
    fps: int = 25,
    data_root: str | None = None,
    split_dir: str | None = None,
    config: dict | None = None,
):
    """YOLO dataset generation matching the working notebook (with deduplication).

    The split is the *canonical* one (1.6). This function used to draw its own
    `GroupShuffleSplit` with its own seed and fraction, so the videos the
    detector trained on and the videos the VAEs trained on were different
    partitions of the same data — and the features were split a third way
    relative to the detector that produced them. The assignment now comes from
    `resolve_split`, i.e. the same `split_manifest.json` every other stage
    reads, which is also where 1.7's plumbing lands: seed and fractions come
    from `config` rather than from module-level defaults a caller can forget
    to override.

    `split_dir` is where that manifest lives — the run's output directory for
    the orchestrator, `output_dir` itself when this is run standalone.
    """
    config = CONFIG if config is None else config
    split_dir = output_dir if split_dir is None else split_dir

    if data_root is None:
        data_root = download_dataset()

    annotations_dir = os.path.join(data_root, "annotations")
    frames_dir = os.path.join(data_root, "rawframes_mini")

    ann_file = os.path.join(annotations_dir, "ava_train_v2.1.csv")
    df = pd.read_csv(ann_file, header=None, dtype={0: str})
    df.columns = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "target_id"]

    # Drawn from the full annotation frame, before the frames-on-disk filter
    # below: a video whose frames are all missing still gets an assignment, so
    # this manifest stays a superset of the row-level one the VAE stages
    # resolve. Filtering first is what 1.4's docstring warns against.
    assignment = resolve_split(split_dir, df, **split_params(config))
    video_to_split = {
        str(video): name for name in SPLIT_NAMES for video in assignment[name]
    }

    valid_video_ids = set(os.listdir(frames_dir))
    df = df[df["video_id"].isin(valid_video_ids)].copy()
    print(f"Raw annotations: {len(df)}")

    boxes_per_image: dict[str, set] = defaultdict(set)
    image_video: dict[str, str] = {}
    image_path: dict[str, str] = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Grouping & dedup"):
        video_id = str(row["video_id"])
        frame_idx = int(row["timestamp"] * fps)
        img_path = Path(frames_dir) / video_id / f"img_{frame_idx:05d}.jpg"

        if not img_path.exists():
            continue

        x1 = max(0.0, min(1.0, row["x1"]))
        x2 = max(0.0, min(1.0, row["x2"]))
        y1 = max(0.0, min(1.0, row["y1"]))
        y2 = max(0.0, min(1.0, row["y2"]))

        if x1 >= x2 or y1 >= y2:
            continue

        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1

        box_key = (
            round(x_center, 6),
            round(y_center, 6),
            round(width, 6),
            round(height, 6),
        )
        image_key = f"{video_id}_{frame_idx:05d}"
        # The video and the source path are recorded per image rather than
        # re-derived from `image_key` with `split("_")[0]` and from an arbitrary
        # element of the box set: video ids may contain "_", and the set's
        # iteration order is not a property of the image (9.8).
        image_video[image_key] = video_id
        image_path[image_key] = str(img_path)
        boxes_per_image[image_key].add(box_key)

    print(f"Unique images with at least one cow: {len(boxes_per_image)}")

    images_by_split: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for image_key in sorted(boxes_per_image):
        images_by_split[video_to_split[image_video[image_key]]].append(image_key)
    print(
        "Videos per split: "
        + ", ".join(f"{name}={len(assignment[name])}" for name in SPLIT_NAMES)
        + " | images: "
        + ", ".join(f"{name}={len(images_by_split[name])}" for name in SPLIT_NAMES)
    )

    for split in SPLIT_NAMES:
        if not images_by_split[split]:
            continue
        os.makedirs(f"{output_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/labels", exist_ok=True)

    for split in SPLIT_NAMES:
        for image_key in tqdm(images_by_split[split], desc=f"Writing {split}"):
            dest_img = Path(output_dir) / split / "images" / f"{image_key}.jpg"
            if not dest_img.exists():
                shutil.copy(image_path[image_key], dest_img)

            label_path = Path(output_dir) / split / "labels" / f"{image_key}.txt"
            with open(label_path, "w") as f:
                for box_key in sorted(boxes_per_image[image_key]):
                    xc, yc, w, h = box_key
                    f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    print(f"Dataset ready at {output_dir}")
    return assignment


if __name__ == "__main__":
    set_seed(CONFIG["random_seed"])
    create_yolo_dataset()
