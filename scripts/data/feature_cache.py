"""Disk-backed cache of YOLO backbone features.

The VAE stages re-extract features from disk on every access. `CowSequenceDataset`
re-instantiates its parent (so `NormalisedSeqDataset` triggers a second full pass)
and then re-extracts per `__getitem__`, which means YOLO runs at least twice per
pipeline invocation and once per frame per epoch thereafter.

This module extracts each distinct crop exactly once and persists the result as a
`(N, D)` array plus its key list and validity mask, keyed by a hash of the YOLO
weights that produced it. Two datasets built over overlapping frames (the flat-VAE
frame set and the LSTM-VAE window set) share the same cache file and each pay only
for the keys they add.
"""

import json
import os
import time

import numpy as np
from tqdm import tqdm

from scripts.utils.hashing import hash_json, sha256_file_or_none
from scripts.utils.io import atomic_save_npy, atomic_write_json

FEATURE_CACHE_FORMAT_VERSION = 1

# Bump when the extraction code itself changes, so old caches are not reused.
EXTRACTOR_VERSION = "1"

DEFAULT_INPUT_SIZE = 224


def frame_image_path(frames_dir, video_id, frame_idx):
    return os.path.join(frames_dir, str(video_id), f"img_{int(frame_idx):05d}.jpg")


def frame_index(timestamp, fps=25):
    return int(timestamp * fps)


def make_feature_key(video_id, frame_idx, bbox, fps=25):
    """Stable identity for one crop.

    The bbox is part of the key even though the current SPPF hook is not
    box-conditioned (Phase 5.1 makes it ROI-aligned, at which point the bbox
    genuinely changes the feature and a bbox-free key would serve wrong vectors).
    """
    x1, y1, x2, y2 = bbox
    return (
        f"{video_id}|{int(frame_idx):05d}|{float(x1):.6f}|{float(y1):.6f}|"
        f"{float(x2):.6f}|{float(y2):.6f}"
    )


def parse_feature_key(key):
    video_id, frame_idx, x1, y1, x2, y2 = key.split("|")
    return video_id, int(frame_idx), (float(x1), float(y1), float(x2), float(y2))


def row_feature_key(row, video_id, fps=25):
    return make_feature_key(
        video_id,
        frame_index(row["timestamp"], fps),
        (row["x1"], row["y1"], row["x2"], row["y2"]),
        fps,
    )


def make_key_extractor(frames_dir, feature_extractor, fps=25, device="cuda", desc="Extracting features"):
    """Build a `key -> feature | None` callable backed by the YOLO backbone.

    cv2 / torchvision / ultralytics are imported lazily inside the closure so
    that importing this module — and therefore the caching logic — does not
    require the vision stack to be installed.
    """

    def extract(key):
        import cv2

        from scripts.models.feature_extractor import extract_cow_features

        video_id, idx, (x1, y1, x2, y2) = parse_feature_key(key)
        img_path = frame_image_path(frames_dir, video_id, idx)
        if not os.path.isfile(img_path):
            return None
        img = cv2.imread(img_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        bbox = (x1 * w, y1 * h, x2 * w, y2 * h)
        return extract_cow_features(img, bbox, feature_extractor, device=device)

    extract.description = desc
    return extract


def cache_signature(weights_sha256, layer_index=9, input_size=DEFAULT_INPUT_SIZE):
    return {
        "format_version": FEATURE_CACHE_FORMAT_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "weights_sha256": weights_sha256,
        "layer_index": int(layer_index),
        "input_size": int(input_size),
    }


class FeatureCache:
    """Keyed feature store persisted next to the run's other artifacts.

    Layout in `directory` (one set per weights file / hook configuration):
        features_<w16>_layer<L>_<size>.npy   (N, D) float32
        features_<w16>_layer<L>_<size>_keys.npy   (N,) str
        features_<w16>_layer<L>_<size>_valid.npy  (N,) bool
        features_<w16>_layer<L>_<size>_meta.json  signature + counts
    """

    def __init__(self, directory, signature):
        self.directory = directory
        self.signature = dict(signature)
        stem = (
            f"features_{str(signature['weights_sha256'])[:16]}"
            f"_layer{signature['layer_index']}_{signature['input_size']}"
        )
        if directory is None:
            # In-memory only: `ensure` still extracts each key once, but nothing
            # is persisted and no paths exist.
            self.features_path = None
            self.keys_path = None
            self.valid_path = None
            self.meta_path = None
        else:
            self.features_path = os.path.join(directory, f"{stem}.npy")
            self.keys_path = os.path.join(directory, f"{stem}_keys.npy")
            self.valid_path = os.path.join(directory, f"{stem}_valid.npy")
            self.meta_path = os.path.join(directory, f"{stem}_meta.json")
        self.signature_hash = hash_json(self.signature)

        self._index = {}
        self._features = None
        self._valid = None
        self._loaded = False

    # ── loading / saving ────────────────────────────────────────────────

    def load(self):
        """Read the cache from disk if present and matching. Missing is fine."""
        if self._loaded:
            return self
        self._loaded = True
        if self.directory is None or not os.path.isfile(self.meta_path):
            return self
        try:
            with open(self.meta_path) as f:
                meta = json.load(f)
            if meta.get("signature_hash") != self.signature_hash:
                raise ValueError("signature mismatch")
            keys = np.load(self.keys_path, allow_pickle=False)
            features = np.load(self.features_path, allow_pickle=False)
            valid = np.load(self.valid_path, allow_pickle=False)
        except (OSError, ValueError, json.JSONDecodeError):
            self._index, self._features, self._valid = {}, None, None
            return self

        if len(keys) != len(features) or len(keys) != len(valid):
            self._index, self._features, self._valid = {}, None, None
            return self

        self._index = {str(k): i for i, k in enumerate(keys)}
        self._features = features
        self._valid = valid.astype(bool)
        return self

    def save(self):
        if self._features is None or self.directory is None:
            return
        os.makedirs(self.directory, exist_ok=True)
        keys = np.empty(len(self._index), dtype=object)
        for key, i in self._index.items():
            keys[i] = key
        keys = np.array([str(k) for k in keys], dtype="U")
        atomic_save_npy(self.keys_path, keys)
        atomic_save_npy(self.features_path, self._features)
        atomic_save_npy(self.valid_path, self._valid.astype(bool))
        atomic_write_json(
            self.meta_path,
            {
                "signature": self.signature,
                "signature_hash": self.signature_hash,
                "n_keys": int(len(self._index)),
                "feature_dim": int(self._features.shape[1]),
                "n_valid": int(self._valid.sum()),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    # ── population ──────────────────────────────────────────────────────

    @property
    def size(self):
        self.load()
        return 0 if self._features is None else len(self._index)

    @property
    def feature_dim(self):
        self.load()
        return None if self._features is None else int(self._features.shape[1])

    def missing(self, keys):
        self.load()
        return [k for k in keys if k not in self._index]

    def ensure(self, keys, extract_fn, verbose=True):
        """Extract and store any of `keys` that are not cached yet.

        Returns the array of row indices into the cache, in the order given, so
        callers can map their own ordering onto cache rows.
        """
        self.load()
        unknown = []
        seen = set()
        for key in keys:
            if key in self._index or key in seen:
                continue
            seen.add(key)
            unknown.append(key)

        if unknown:
            if extract_fn is None:
                raise ValueError(
                    f"{len(unknown)} feature(s) missing from the cache and no "
                    "feature_extractor was supplied to compute them"
                )
            desc = getattr(extract_fn, "description", "Extracting features")
            iterator = tqdm(unknown, desc=desc) if verbose else unknown
            extracted = []
            for key in iterator:
                feat = extract_fn(key)
                extracted.append(feat)

            dim = next((len(f) for f in extracted if f is not None), None)
            if dim is None:
                raise RuntimeError(
                    "feature extraction produced nothing — check the frames directory "
                    "and the YOLO weights"
                )

            new_features = np.zeros((len(unknown), dim), dtype=np.float32)
            new_valid = np.zeros(len(unknown), dtype=bool)
            for i, feat in enumerate(extracted):
                if feat is None:
                    continue
                new_features[i] = np.asarray(feat, dtype=np.float32).ravel()
                new_valid[i] = True

            if self._features is None:
                self._features = new_features
                self._valid = new_valid
            else:
                if new_features.shape[1] != self._features.shape[1]:
                    raise ValueError(
                        "feature dim changed mid-cache "
                        f"({self._features.shape[1]} -> {new_features.shape[1]})"
                    )
                self._features = np.concatenate([self._features, new_features], axis=0)
                self._valid = np.concatenate([self._valid, new_valid], axis=0)

            base = len(self._index)
            for i, key in enumerate(unknown):
                self._index[key] = base + i

            self.save()

        return np.asarray([self._index[k] for k in keys], dtype=np.int64)

    # ── reading ─────────────────────────────────────────────────────────

    def rows(self, indices):
        """Features for the given cache rows, shape (len(indices), D)."""
        self.load()
        return self._features[np.asarray(indices, dtype=np.int64)]

    def is_valid(self, indices):
        self.load()
        return self._valid[np.asarray(indices, dtype=np.int64)]

    def contains(self, key):
        self.load()
        return key in self._index


FEATURE_CACHE_DIRNAME = "feature_cache"


def open_feature_cache(cache_dir, weights_path, layer_index=9, input_size=DEFAULT_INPUT_SIZE):
    """Open (or create) the cache for a given YOLO weights file."""
    weights_sha = sha256_file_or_none(weights_path)
    if weights_sha is None:
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    signature = cache_signature(weights_sha, layer_index=layer_index, input_size=input_size)
    return FeatureCache(cache_dir, signature)


def open_run_feature_cache(output_dir, weights_path, layer_index=9, input_size=DEFAULT_INPUT_SIZE):
    """Open the cache belonging to a pipeline run's output directory."""
    return open_feature_cache(
        os.path.join(output_dir, FEATURE_CACHE_DIRNAME),
        weights_path,
        layer_index=layer_index,
        input_size=input_size,
    )
