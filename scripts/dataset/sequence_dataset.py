import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.models.feature_extractor import extract_cow_features


class CowSequenceDataset(Dataset):
    def __init__(
        self,
        df,
        frames_dir,
        feature_extractor,
        seq_len=16,
        stride=1,
        normal_action_ids=None,
        fps=25,
        device="cuda",
    ):
        self.df = df
        self.frames_dir = frames_dir
        self.feature_extractor = feature_extractor
        self.seq_len = seq_len
        self.stride = stride
        self.fps = fps
        self.device = device
        self.normal_action_ids = set(normal_action_ids or [])

        self.tracks = defaultdict(list)
        for idx, row in df.iterrows():
            if self.normal_action_ids and row["action_id"] not in self.normal_action_ids:
                continue
            key = (str(row["video_id"]), row["target_id"])
            self.tracks[key].append(
                (row["timestamp"], row["x1"], row["y1"], row["x2"], row["y2"])
            )

        for key in self.tracks:
            self.tracks[key].sort(key=lambda x: x[0])

        self.sequences = []
        for key, frames in self.tracks.items():
            n_frames = len(frames)
            for start in range(0, n_frames - seq_len + 1, stride):
                self.sequences.append((key, start))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        key, start = self.sequences[idx]
        frames = self.tracks[key]
        video_id, target_id = key
        seq_features = []
        for i in range(start, start + self.seq_len):
            ts, x1, y1, x2, y2 = frames[i]
            frame_idx = int(ts * self.fps)
            img_path = os.path.join(
                self.frames_dir, video_id, f"img_{frame_idx:05d}.jpg"
            )
            img = cv2.imread(img_path)
            if img is None:
                feat = np.zeros(256, dtype=np.float32)
            else:
                h, w = img.shape[:2]
                bbox = (x1 * w, y1 * h, x2 * w, y2 * h)
                feat = extract_cow_features(
                    img, bbox, self.feature_extractor, device=self.device
                )
                if feat is None:
                    feat = np.zeros(256, dtype=np.float32)
            seq_features.append(feat)
        seq_features = np.stack(seq_features, axis=0)
        return torch.tensor(seq_features, dtype=torch.float32)


class NormalisedSeqDataset(CowSequenceDataset):
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __getitem__(self, idx):
        seq = super().__getitem__(idx)
        seq = (seq - self.mean) / self.std
        return seq
