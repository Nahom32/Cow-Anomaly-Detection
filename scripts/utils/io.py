"""Atomic file writes.

A crash in the middle of `torch.save` / `np.save` leaves a truncated file that
still exists on disk, and existence-only step checks (`is_step_done`) then report
the step as complete. Every writer in this repo goes through here so a file is
either fully there or not there at all.
"""

import contextlib
import json
import os
import tempfile


def _fsync_path(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


@contextlib.contextmanager
def atomic_path(final_path):
    """Yield a temp path next to `final_path`, renamed onto it on clean exit.

    The temp file lives in the destination directory so `os.replace` is a
    same-filesystem rename (atomic) rather than a copy.
    """
    final_path = os.fspath(final_path)
    directory = os.path.dirname(os.path.abspath(final_path)) or "."
    os.makedirs(directory, exist_ok=True)
    prefix = os.path.basename(final_path) + "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=prefix, suffix=".tmp")
    os.close(fd)
    try:
        yield tmp_path
        _fsync_path(tmp_path)
        os.replace(tmp_path, final_path)
        _fsync_path(directory)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def atomic_write_bytes(path, data):
    with atomic_path(path) as tmp:
        with open(tmp, "wb") as f:
            f.write(data)


def atomic_write_text(path, text):
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path, obj, indent=2):
    atomic_write_text(path, json.dumps(obj, indent=indent, sort_keys=True, default=str) + "\n")


def atomic_savez(path, **arrays):
    import numpy as np

    with atomic_path(path) as tmp:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)


def atomic_save_npy(path, array):
    import numpy as np

    with atomic_path(path) as tmp:
        with open(tmp, "wb") as f:
            np.save(f, array)
