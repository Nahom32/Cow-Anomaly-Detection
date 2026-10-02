import numpy as np

from scripts.data.feature_cache import (
    DEFAULT_INPUT_SIZE,
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
    """Frame-level features for every normal-action annotation row, plus their videos.

    Returns `(features, video_ids)`: a `(n_rows, dim)` matrix and the `video_id` of
    each row, in the same order. The ids are load-bearing, not a convenience — a
    split can only be grouped (1.3) if the caller can tell which video each row came
    from, and returning a bare matrix is what forced the split down to row level.

    When `feature_cache` is given, rows already in it are read from disk and only
    the missing crops go through YOLO. Without one, an in-memory cache is used,
    which still extracts each distinct crop once.
    """
    if normal_action_ids is None:
        normal_action_ids = [0, 1, 2]

    keys = []
    video_ids = []
    for _, row in df.iterrows():
        if row["action_id"] not in normal_action_ids:
            continue
        keys.append(row_feature_key(row, str(row["video_id"]), fps))
        video_ids.append(str(row["video_id"]))

    if not keys:
        raise RuntimeError("no normal-action rows to extract features from")

    if feature_cache is None:
        feature_cache = FeatureCache(directory=None, signature=cache_signature("uncached"))

    extract_fn = None
    if feature_extractor is not None:
        # Take the resize from the cache signature: a crop extracted at a
        # different size than its cache key would make the two disagree.
        extract_fn = make_key_extractor(
            frames_dir, feature_extractor, fps=fps, device=device,
            desc="Extracting frame features",
            input_size=int(feature_cache.signature.get("input_size", DEFAULT_INPUT_SIZE)),
        )

    rows = feature_cache.ensure(keys, extract_fn, verbose=verbose)
    valid = feature_cache.is_valid(rows)
    features = feature_cache.rows(rows)[valid]
    # `valid` drops rows whose crop could not be extracted, so the ids must be
    # filtered by the same mask or they would no longer line up with `features`.
    kept_videos = np.array([video_ids[i] for i in np.flatnonzero(valid)], dtype=object)
    if kept_videos.shape[0] != features.shape[0]:
        raise RuntimeError(
            f"video id count ({kept_videos.shape[0]}) does not match feature rows "
            f"({features.shape[0]}); a grouped split over misaligned ids would silently "
            "put the wrong video's frames on each side"
        )
    return np.ascontiguousarray(features, dtype=np.float32), kept_videos
