import os

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from scripts.data.download_dataset import download_dataset


def explore_dataset(data_root: str | None = None):
    if data_root is None:
        data_root = download_dataset()

    print(f"\nDataset structure preview:")
    for root, dirs, files in os.walk(data_root):
        if files:
            rel = os.path.relpath(root, data_root)
            print(f"  {rel}/  -> {len(files)} files (first 3: {files[:3]})")

    annotations_dir = os.path.join(data_root, "annotations")
    frames_dir = os.path.join(data_root, "rawframes_mini")

    ann_file = os.path.join(annotations_dir, "ava_train_v2.1.csv")
    if not os.path.exists(ann_file):
        for f in os.listdir(annotations_dir):
            if f.endswith(".csv"):
                ann_file = os.path.join(annotations_dir, f)
                break
    print(f"\nUsing annotations: {ann_file}")

    df = pd.read_csv(ann_file, header=None, dtype={0: str})
    df.columns = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "target_id"]

    print(f"Total samples: {len(df)}")
    print(df.head())

    valid_ids = set(os.listdir(frames_dir))
    df = df[df["video_id"].isin(valid_ids)]
    print(f"Filtered dataset size: {len(df)}")
    print(f"Number of unique video IDs: {df['video_id'].nunique()}")
    print(f"Number of unique action IDs: {df['action_id'].nunique()}")

    return df, data_root


if __name__ == "__main__":
    explore_dataset()
