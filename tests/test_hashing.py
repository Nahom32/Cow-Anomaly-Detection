import hashlib

import numpy as np

from scripts.utils.hashing import (
    canonical_json,
    hash_array,
    hash_json,
    sha256_bytes,
    sha256_file,
    sha256_file_or_none,
)


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "weights.pt"
    payload = b"yolo" * 10_000
    path.write_bytes(payload)
    assert sha256_file(str(path)) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_or_none_swallows_missing_files(tmp_path):
    assert sha256_file_or_none(str(tmp_path / "nope.pt")) is None


def test_sha256_bytes_matches_hashlib():
    assert sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_hash_json_ignores_key_order():
    assert hash_json({"a": 1, "b": 2}) == hash_json({"b": 2, "a": 1})
    assert hash_json({"a": [1, {"x": 1}]}) == hash_json({"a": [1, {"x": 1}]})


def test_hash_json_detects_value_changes():
    assert hash_json({"a": 1}) != hash_json({"a": 2})
    assert hash_json({"a": 1}) != hash_json({"a": 1, "b": None})


def test_canonical_json_is_compact_and_sorted():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_hash_array_detects_value_changes():
    a = np.arange(6, dtype=np.float32)
    assert hash_array(a) != hash_array(a + 1e-6)


def test_hash_array_detects_dtype_and_shape_changes():
    a = np.arange(6, dtype=np.float32)
    assert hash_array(a) != hash_array(a.astype(np.float64)), "dtype is part of the identity"
    assert hash_array(a) != hash_array(a.reshape(2, 3)), "shape is part of the identity"


def test_hash_array_is_stable_across_memory_layout():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    fortran = np.asfortranarray(a)
    assert not fortran.flags["C_CONTIGUOUS"]
    assert fortran.tolist() == a.tolist()
    assert hash_array(a) == hash_array(fortran)


def test_config_hash_changes_when_config_changes():
    from scripts.config import CONFIG

    baseline = hash_json(CONFIG)
    assert hash_json(dict(CONFIG)) == baseline, "copying the config must not change its hash"
    assert hash_json({**CONFIG, "random_seed": CONFIG["random_seed"] + 1}) != baseline
