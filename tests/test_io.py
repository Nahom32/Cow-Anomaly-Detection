import json
import os

import numpy as np
import pytest

from scripts.utils.io import (
    atomic_path,
    atomic_save_npy,
    atomic_savez,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
)


def temp_files_in(directory):
    return [f for f in os.listdir(directory) if f.endswith(".tmp")]


def test_atomic_write_json_roundtrip(tmp_path):
    path = tmp_path / "out" / "run_manifest.json"
    payload = {"b": 2, "a": [1, 2, 3], "nested": {"x": None}}
    atomic_write_json(path, payload)

    assert json.loads(path.read_text()) == payload
    assert temp_files_in(path.parent) == []


def test_atomic_write_json_is_sorted_and_ends_with_newline(tmp_path):
    path = tmp_path / "m.json"
    atomic_write_json(path, {"z": 1, "a": 2})
    text = path.read_text()
    assert text.index('"a"') < text.index('"z"')
    assert text.endswith("\n")


def test_failed_write_leaves_the_original_intact(tmp_path):
    path = tmp_path / "data.json"
    atomic_write_json(path, {"generation": 1})

    # Simulate a crash partway through the second write.
    with pytest.raises(KeyboardInterrupt):
        with atomic_path(path) as tmp:
            with open(tmp, "w") as f:
                f.write('{"generation": 2')
                f.flush()
            raise KeyboardInterrupt

    assert json.loads(path.read_text()) == {"generation": 1}, "interrupted write must not land"
    assert temp_files_in(tmp_path) == []


def test_failed_write_of_a_new_file_leaves_nothing(tmp_path):
    target = tmp_path / "new.json"

    with pytest.raises(RuntimeError):
        with atomic_path(target) as tmp:
            with open(tmp, "w") as f:
                f.write("partial")
            raise RuntimeError("crash mid-write")

    assert not target.exists()
    assert temp_files_in(tmp_path) == []


def test_atomic_path_replaces_on_success(tmp_path):
    target = tmp_path / "file.bin"
    with atomic_path(target) as tmp:
        with open(tmp, "wb") as f:
            f.write(b"new contents")
        assert not target.exists(), "target must not appear until the write completes"
    assert target.read_bytes() == b"new contents"


def test_atomic_save_npy_roundtrip(tmp_path):
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "features.npy"
    atomic_save_npy(path, array)

    assert np.array_equal(np.load(path), array)
    assert path.suffix == ".npy"
    assert temp_files_in(tmp_path) == []


def test_atomic_savez_roundtrip(tmp_path):
    path = tmp_path / "bundle.npz"
    atomic_savez(path, a=np.arange(3), b=np.ones((2, 2)))

    with np.load(path) as data:
        assert data["a"].tolist() == [0, 1, 2]
        assert data["b"].shape == (2, 2)
    assert temp_files_in(tmp_path) == []


def test_atomic_write_bytes_and_text(tmp_path):
    binary = tmp_path / "blob.bin"
    text = tmp_path / "note.txt"
    atomic_write_bytes(binary, b"\x00\x01\x02")
    atomic_write_text(text, "hello")

    assert binary.read_bytes() == b"\x00\x01\x02"
    assert text.read_text() == "hello"


def test_atomic_path_creates_missing_directories(tmp_path):
    target = tmp_path / "a" / "b" / "c.json"
    atomic_write_json(target, {"ok": True})
    assert target.exists()
