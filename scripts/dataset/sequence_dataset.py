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
        noise_std=0.05,
        device="cuda",
    ):
        self.df = df
        self.frames_dir = frames_dir
        self.feature_extractor = feature_extractor
        self.seq_len = seq_len
        self.stride = stride
        self.fps = fps
        self.noise_std = noise_std
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

    def _load_feat(self, video_id, ts, x1, y1, x2, y2):
        frame_idx = int(ts * self.fps)
        img_path = os.path.join(
            self.frames_dir, video_id, f"img_{frame_idx:05d}.jpg"
        )
        if not os.path.isfile(img_path):
            return None
        img = cv2.imread(img_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        bbox = (x1 * w, y1 * h, x2 * w, y2 * h)
        return extract_cow_features(
            img, bbox, self.feature_extractor, device=self.device
        )

    def __getitem__(self, idx):
        key, start = self.sequences[idx]
        frames = self.tracks[key]
        video_id, target_id = key

        # Extract all features for this sequence, tracking which ones succeeded
        seq_feats = []
        valid_mask = []
        for i in range(start, start + self.seq_len):
            ts, x1, y1, x2, y2 = frames[i]
            feat = self._load_feat(video_id, ts, x1, y1, x2, y2)
            if feat is not None:
                seq_feats.append(feat)
                valid_mask.append(True)
            else:
                seq_feats.append(None)
                valid_mask.append(False)

        # Fill gaps: prefer previous frame + noise, fall back to next, then zeros
        filled = []
        prev_feat = None
        for i, (feat, valid) in enumerate(zip(seq_feats, valid_mask)):
            if valid:
                prev_feat = feat
                filled.append(feat)
            elif prev_feat is not None:
                # Previous frame + small noise for temporal smoothness
                filled.append(prev_feat + np.random.randn(*prev_feat.shape).astype(np.float32) * self.noise_std)
            else:
                # First frame(s) missing — try next valid frame + noise
                next_feat = None
                for j in range(i + 1, len(seq_feats)):
                    if valid_mask[j]:
                        next_feat = seq_feats[j]
                        break
                if next_feat is not None:
                    filled.append(next_feat + np.random.randn(*next_feat.shape).astype(np.float32) * self.noise_std)
                else:
                    # Entire sequence is bad — zeros as last resort
                    filled.append(np.zeros(256, dtype=np.float32))

        seq_features = np.stack(filled, axis=0)
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
