import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from scripts.data.download_dataset import download_dataset


def create_yolo_dataset_notebook1(
    output_dir: str = "/content/cow_detection_dataset",
    fps: int = 25,
    val_ratio: float = 0.2,
    data_root: str | None = None,
) -> pd.DataFrame:
    """YOLO dataset generation from notebook 1 (without deduplication)."""
    if data_root is None:
        data_root = download_dataset()

    annotations_dir = os.path.join(data_root, "miniannotations")
    frames_dir = os.path.join(data_root, "rawframes_mini")

    ann_file = os.path.join(annotations_dir, "ava_train_v2.1.csv")
    df = pd.read_csv(ann_file, header=None, dtype={0: str})
    df.columns = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "target_id"]

    valid_ids = set(os.listdir(frames_dir))
    df["video_id"] = df["video_id"].apply(lambda x: str(int(x)))
    df = df[df["video_id"].isin(valid_ids)]

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:
        os.makedirs(f"{output_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/labels", exist_ok=True)

    video_ids = df["video_id"].unique()
    np.random.seed(42)
    val_videos = set(
        np.random.choice(video_ids, size=int(val_ratio * len(video_ids)), replace=False)
    )

    valid_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_id = row["video_id"]
        frame_idx = int(row["timestamp"] * fps)
        img_filename = f"img_{frame_idx:05d}.jpg"
        img_path = Path(frames_dir) / video_id / img_filename

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        x1 = max(0.0, min(1.0, row["x1"]))
        x2 = max(0.0, min(1.0, row["x2"]))
        y1 = max(0.0, min(1.0, row["y1"]))
        y2 = max(0.0, min(1.0, row["y2"]))

        if x1 >= x2 or y1 >= y2:
            continue

        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        box_w = x2 - x1
        box_h = y2 - y1

        yolo_line = f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n"

        split = "val" if video_id in val_videos else "train"
        img_dest = Path(output_dir) / split / "images" / f"{video_id}_{frame_idx:05d}.jpg"
        cv2.imwrite(str(img_dest), img)

        label_dest = Path(output_dir) / split / "labels" / f"{video_id}_{frame_idx:05d}.txt"
        with open(label_dest, "w") as f:
            f.write(yolo_line)

        valid_count += 1

    print(f"Created {valid_count} valid entries at {output_dir}")
    return df


def create_yolo_dataset_notebook2(
    output_dir: str = "/content/cow_detection_dataset_fixed",
    fps: int = 25,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    data_root: str | None = None,
):
    """YOLO dataset generation from notebook 2 (with deduplication)."""
    if data_root is None:
        data_root = download_dataset()

    annotations_dir = os.path.join(data_root, "annotations")
    frames_dir = os.path.join(data_root, "rawframes_mini")

    ann_file = os.path.join(annotations_dir, "ava_train_v2.1.csv")
    df = pd.read_csv(ann_file, header=None, dtype={0: str})
    df.columns = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "target_id"]

    valid_video_ids = set(os.listdir(frames_dir))
    df = df[df["video_id"].isin(valid_video_ids)].copy()
    print(f"Raw annotations: {len(df)}")

    boxes_per_image: dict[str, set] = defaultdict(set)

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
        boxes_per_image[image_key].add((str(img_path), box_key))

    print(f"Unique images with at least one cow: {len(boxes_per_image)}")

    all_video_ids = list(set(key.split("_")[0] for key in boxes_per_image.keys()))
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=1 - train_ratio, random_state=random_seed
    )
    train_idx, val_idx = next(splitter.split(all_video_ids, groups=all_video_ids))

    train_videos = set([all_video_ids[i] for i in train_idx])
    val_videos = set([all_video_ids[i] for i in val_idx])
    print(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")

    for split in ["train", "val"]:
        os.makedirs(f"{output_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/labels", exist_ok=True)

    for image_key, box_set in tqdm(boxes_per_image.items(), desc="Writing dataset"):
        video_id = image_key.split("_")[0]
        split = "val" if video_id in val_videos else "train"

        img_path = next(iter(box_set))[0]
        dest_img = Path(output_dir) / split / "images" / f"{image_key}.jpg"
        if not dest_img.exists():
            shutil.copy(img_path, dest_img)

        label_path = Path(output_dir) / split / "labels" / f"{image_key}.txt"
        with open(label_path, "w") as f:
            for _, box_key in box_set:
                xc, yc, w, h = box_key
                f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    print(f"Dataset ready at {output_dir}")


if __name__ == "__main__":
    create_yolo_dataset_notebook1()
