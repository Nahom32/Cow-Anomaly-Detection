import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CowDatasetHybrid(Dataset):
    """PyTorch Dataset for anomaly detection using normal/anomaly action pairs."""

    def __init__(
        self,
        df,
        frames_dir: str,
        normal_classes: list[int],
        num_frames: int = 8,
        fps: int = 25,
    ):
        self.df = df.reset_index(drop=True)
        self.frames_dir = frames_dir
        self.num_frames = num_frames
        self.fps = fps

        self.normal_df = self.df[
            self.df["action_id"].isin([c + 1 for c in normal_classes])
        ]
        self.anomaly_df = self.df[
            ~self.df["action_id"].isin([c + 1 for c in normal_classes])
        ]

    def __len__(self):
        return len(self.normal_df)

    def _load_clip(self, row):
        video_id = str(int(row.video_id))
        video_folder = os.path.join(self.frames_dir, video_id)

        if not os.path.exists(video_folder):
            return torch.zeros((self.num_frames, 3, 224, 224))

        frame_files = sorted(os.listdir(video_folder))
        if len(frame_files) == 0:
            return torch.zeros((self.num_frames, 3, 224, 224))

        center_idx = int(row.timestamp * self.fps)
        center_idx = min(max(center_idx, 0), len(frame_files) - 1)

        start = max(0, center_idx - self.num_frames // 2)
        selected = frame_files[start : start + self.num_frames]

        frames = []
        for f in selected:
            img = cv2.imread(os.path.join(video_folder, f))
            if img is None:
                continue

            h, w, _ = img.shape
            x1 = int(row.x1 * w)
            y1 = int(row.y1 * h)
            x2 = int(row.x2 * w)
            y2 = int(row.y2 * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            img = img[y1:y2, x1:x2]
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = (img - 0.5) / 0.5
            frames.append(img)

        if len(frames) == 0:
            frames = [np.zeros((224, 224, 3))] * self.num_frames

        if len(frames) < self.num_frames:
            frames += [frames[-1]] * (self.num_frames - len(frames))

        frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
        return frames

    def __getitem__(self, idx):
        normal_row = self.normal_df.iloc[idx]
        normal_clip = self._load_clip(normal_row)

        pseudo_row = self.anomaly_df.sample(1).iloc[0]
        pseudo_clip = self._load_clip(pseudo_row)

        return normal_clip, pseudo_clip
