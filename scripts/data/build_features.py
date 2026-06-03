import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.models.feature_extractor import extract_cow_features


def build_feature_dataset(
    df,
    frames_dir,
    feature_extractor,
    normal_action_ids=None,
    fps=25,
    device="cuda",
):
    if normal_action_ids is None:
        normal_action_ids = [0, 1, 2]

    features = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if row["action_id"] not in normal_action_ids:
            continue
        video_id = str(row["video_id"])
        frame_idx = int(row["timestamp"] * fps)
        img_path = os.path.join(frames_dir, video_id, f"img_{frame_idx:05d}.jpg")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue

        h_img, w_img, _ = img.shape
        bbox_abs = (
            row["x1"] * w_img,
            row["y1"] * h_img,
            row["x2"] * w_img,
            row["y2"] * h_img,
        )

        feat = extract_cow_features(img, bbox_abs, feature_extractor, device=device)
        if feat is not None:
            features.append(feat)

    return np.array(features)


def normalize_features(features, eps=1e-8):
    min_val = features.min(axis=0)
    max_val = features.max(axis=0)
    normalized = (features - min_val) / (max_val - min_val + eps)
    return normalized, min_val, max_val
