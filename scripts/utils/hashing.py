"""Content hashes used to bind artifacts to the inputs that produced them."""

import hashlib
import json

_CHUNK = 1 << 20


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    """sha256 of a file's bytes, streamed so a multi-GB weights file is fine."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file_or_none(path):
    try:
        return sha256_file(path)
    except OSError:
        return None


def canonical_json(obj):
    """Stable JSON: sorted keys, no insignificant whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def hash_json(obj):
    """Hash of any JSON-serialisable object, stable across key orderings."""
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def hash_array(array):
    """Hash of an array's dtype, shape and raw bytes."""
    import numpy as np

    arr = np.ascontiguousarray(array)
    header = f"{arr.dtype.str}|{arr.shape}".encode("utf-8")
    h = hashlib.sha256()
    h.update(header)
    h.update(arr.tobytes())
    return h.hexdigest()
