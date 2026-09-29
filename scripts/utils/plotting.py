"""Matplotlib setup that cannot block a server, a CI job or a queue.

`plt.show()` defaults to blocking until the window is closed. Called from
`run_full_pipeline.py` on a headless machine, matplotlib either blocks forever or
emits a backend warning and the figure is lost. Everything that plots here goes
through `setup_matplotlib` + `show_or_close`, and every figure can be written to
disk instead of displayed.
"""

import os
import sys

import matplotlib

NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "ps", "svg", "template", "cairo"}


def is_headless():
    """True when there is no display to draw on."""
    if os.environ.get("MPLBACKEND"):
        return matplotlib.get_backend().lower() in NON_INTERACTIVE_BACKENDS
    if sys.platform.startswith("linux"):
        return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return False


def setup_matplotlib(headless=None):
    """Force a non-interactive backend when there is no display.

    Returns the backend in use so callers can log it.
    """
    if headless is None:
        headless = is_headless()
    if headless:
        matplotlib.use("Agg", force=True)
    return matplotlib.get_backend()


def is_interactive_backend():
    return matplotlib.get_backend().lower() not in NON_INTERACTIVE_BACKENDS


def show_or_close(fig, save_path=None, dpi=120):
    """Save if asked, show without blocking if the backend can show, then close.

    Closing unconditionally is what keeps repeated training runs from
    accumulating figures until matplotlib warns about too many open figures.
    """
    import matplotlib.pyplot as plt

    if save_path is not None:
        directory = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if is_interactive_backend():
        plt.show(block=False)

    plt.close(fig)
    return save_path
