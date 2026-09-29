import numpy as np

from scripts.data.feature_cache import (
    FeatureCache,
    cache_signature,
    make_key_extractor,
    row_feature_key,
)


def build_feature_dataset(
    df,
    frames_dir,
    feature_extractor=None,
    normal_action_ids=None,
    fps=25,
    device="cuda",
    feature_cache=None,
    verbose=True,
):
    """Frame-level features for every normal-action annotation row.

    When `feature_cache` is given, rows already in it are read from disk and only
    the missing crops go through YOLO. Without one, an in-memory cache is used,
    which still extracts each distinct crop once.
    """
    if normal_action_ids is None:
        normal_action_ids = [0, 1, 2]

    keys = []
    for _, row in df.iterrows():
        if row["action_id"] not in normal_action_ids:
            continue
        keys.append(row_feature_key(row, str(row["video_id"]), fps))

    if not keys:
        raise RuntimeError("no normal-action rows to extract features from")

    if feature_cache is None:
        feature_cache = FeatureCache(directory=None, signature=cache_signature("uncached"))

    extract_fn = None
    if feature_extractor is not None:
        extract_fn = make_key_extractor(
            frames_dir, feature_extractor, fps=fps, device=device,
            desc="Extracting frame features",
        )

    rows = feature_cache.ensure(keys, extract_fn, verbose=verbose)
    valid = feature_cache.is_valid(rows)
    features = feature_cache.rows(rows)[valid]
    return np.ascontiguousarray(features, dtype=np.float32)


def normalize_features(features, eps=1e-8):
    min_val = features.min(axis=0)
    max_val = features.max(axis=0)
    normalized = (features - min_val) / (max_val - min_val + eps)
    return normalized, min_val, max_val
