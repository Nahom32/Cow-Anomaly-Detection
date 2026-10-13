from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.data.feature_cache import (
    DEFAULT_INPUT_SIZE,
    FeatureCache,
    cache_signature,
    make_key_extractor,
    row_feature_key,
)
from scripts.utils.seeding import DEFAULT_SEED, rng_for


def frame_feature_key(frame, video_id, fps=25):
    """Cache key for one `(timestamp, x1, y1, x2, y2)` row of a track."""
    ts, x1, y1, x2, y2 = frame
    return row_feature_key(
        {"timestamp": ts, "x1": x1, "y1": y1, "x2": x2, "y2": y2},
        video_id,
        fps=fps,
    )


class CowSequenceDataset(Dataset):
    """Temporal windows of per-cow features.

    Features come from a `FeatureCache`, so every crop is extracted at most once
    per run no matter how many times a window is read or how many times this
    dataset is re-instantiated (which `NormalisedSeqDataset` used to do).
    """

    def __init__(
        self,
        df,
        frames_dir,
        feature_extractor=None,
        feature_cache=None,
        seq_len=16,
        stride=1,
        normal_action_ids=None,
        fps=25,
        noise_std=0.05,
        device="cuda",
        seed=DEFAULT_SEED,
        verbose=True,
    ):
        self.df = df
        self.frames_dir = frames_dir
        self.feature_extractor = feature_extractor
        self.seq_len = seq_len
        self.stride = stride
        self.fps = fps
        self.noise_std = noise_std
        self.device = device
        self.seed = seed
        self.verbose = verbose
        self.normal_action_ids = set(normal_action_ids or [])

        self.tracks = defaultdict(list)
        for _, row in df.iterrows():
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

        self.feature_cache = self._resolve_cache(feature_cache, verbose=verbose)
        # The feature width comes from the extracted array, not a hardcoded 256
        # (2.12). `_resolve_sequence_rows` populates the cache, so read the width
        # only after it: a cold cache has no dim yet, and falling back to a
        # constant would make the model width depend on a guess instead of the
        # data the backbone actually produced.
        self.sequence_rows = self._resolve_sequence_rows()
        self.feature_dim = self.feature_cache.feature_dim
        if self.feature_dim is None:
            raise RuntimeError(
                "the feature cache is empty, so the feature dimension cannot be "
                "derived from the data (2.12)"
            )

    def __len__(self):
        return len(self.sequences)

    def stack(self, indices):
        """The given windows as one `(n_windows, seq_len, feature_dim)` array.

        Fitting a normalizer on the training split must not require reading the
        validation windows (1.2), so the indices passed here are the ones the
        split selected.
        """
        if not indices:
            raise ValueError("stack() needs at least one window index")
        return np.stack([self[int(i)].numpy() for i in indices], axis=0)

    # ── feature resolution ──────────────────────────────────────────────

    def _resolve_cache(self, feature_cache, verbose=True):
        if feature_cache is not None:
            return feature_cache.load()
        if self.feature_extractor is None:
            raise ValueError(
                "CowSequenceDataset needs either a feature_cache or a feature_extractor"
            )
        cache = FeatureCache(directory=None, signature=cache_signature("uncached"))
        return cache.load()

    def _resolve_sequence_rows(self):
        """Map every (track, window) to rows of the cache, once, up front."""
        extract_fn = None
        if self.feature_extractor is not None:
            extract_fn = make_key_extractor(
                self.frames_dir,
                self.feature_extractor,
                fps=self.fps,
                device=self.device,
                desc="Extracting sequence features",
                input_size=int(self.feature_cache.signature.get("input_size", DEFAULT_INPUT_SIZE)),
            )

        track_rows = {}
        all_keys = []
        for key, frames in self.tracks.items():
            video_id, _ = key
            keys = [frame_feature_key(row, video_id, self.fps) for row in frames]
            track_rows[key] = keys
            all_keys.extend(keys)

        rows = self.feature_cache.ensure(all_keys, extract_fn, verbose=self.verbose)
        # `ensure` returns rows in request order, so re-split per track.
        offset = 0
        resolved = {}
        for key, keys in track_rows.items():
            resolved[key] = rows[offset : offset + len(keys)]
            offset += len(keys)

        return [resolved[key][start : start + self.seq_len] for key, start in self.sequences]

    # ── item access ─────────────────────────────────────────────────────

    def __getitem__(self, idx):
        key, start = self.sequences[idx]
        video_id, target_id = key

        rows = self.sequence_rows[idx]
        feats = self.feature_cache.rows(rows)
        valid_mask = self.feature_cache.is_valid(rows)
        seq_feats = [f if v else None for f, v in zip(feats, valid_mask)]

        # The fill noise is derived from (seed, video, track, window start) rather
        # than drawn from global state, so the same window yields the same filled
        # sequence on every epoch and in every process. Resampling from the global
        # RNG made the val set stochastic and the train set order-dependent.
        rng = rng_for(self.seed, video_id, target_id, start, self.noise_std)

        # Fill gaps: prefer previous frame + noise, fall back to next, then zeros
        filled = []
        prev_feat = None
        for i, (feat, valid) in enumerate(zip(seq_feats, valid_mask)):
            if valid:
                prev_feat = feat
                filled.append(feat)
            elif prev_feat is not None:
                # Previous frame + small noise for temporal smoothness
                filled.append(prev_feat + (rng.standard_normal(prev_feat.shape) * self.noise_std).astype(np.float32))
            else:
                # First frame(s) missing — try next valid frame + noise
                next_feat = None
                for j in range(i + 1, len(seq_feats)):
                    if valid_mask[j]:
                        next_feat = seq_feats[j]
                        break
                if next_feat is not None:
                    filled.append(next_feat + (rng.standard_normal(next_feat.shape) * self.noise_std).astype(np.float32))
                else:
                    # Entire sequence is bad — zeros as last resort
                    filled.append(np.zeros(self.feature_dim, dtype=np.float32))

        seq_features = np.stack(filled, axis=0)
        return torch.tensor(seq_features, dtype=torch.float32)


class NormalisedSeqDataset(CowSequenceDataset):
    """Scales each window with a normalizer the caller fitted on train only.

    `normalizer` arrives as a constructor argument precisely because this class
    must never derive it itself (task 1.2): a dataset that computed its own
    statistics would fit on whatever it was handed — every sequence in the run,
    val windows included — and then normalize each of those windows with numbers
    they helped produce. It is a shared `_FittedNormalizer` (task 2.4), so the
    LSTM-VAE min-max-scales windows with the same object the flat VAE uses on
    rows and both decoders can end in `Sigmoid`.
    """

    def __init__(self, normalizer, **kwargs):
        super().__init__(**kwargs)
        self.normalizer = normalizer

    def __getitem__(self, idx):
        seq = super().__getitem__(idx).numpy()
        return torch.tensor(self.normalizer.transform(seq), dtype=torch.float32)
