"""Dependency and packaging invariants.

These are cheap checks that catch the exact class of drift Phase 0 exists to
prevent: a second dependency list, an unpinned package, a dead dependency.
"""

import os
import re

import pytest

from scripts.setup import MIN_PYTHON, check_python_version, read_requirements

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIREMENTS_PATH = os.path.join(REPO_ROOT, "requirements.txt")

DEAD_DEPENDENCIES = ["decord", "pytorchvideo"]


@pytest.fixture(scope="module")
def requirements():
    return read_requirements(REQUIREMENTS_PATH)


def test_setup_reads_the_requirements_file(requirements):
    with open(REQUIREMENTS_PATH) as f:
        raw_lines = [line for line in f if line.strip() and not line.strip().startswith("#")]
    assert len(requirements) == len(raw_lines), "setup.py must parse every declared requirement"


def test_no_requirement_is_unpinned(requirements):
    unpinned = [r for r in requirements if not re.search(r"(==|>=|<=|~=)", r)]
    assert not unpinned, f"unpinned requirements: {unpinned}"


def test_no_dead_dependencies(requirements):
    for dead in DEAD_DEPENDENCIES:
        assert not any(r.split("=")[0].split(">")[0].split("<")[0].strip() == dead for r in requirements), (
            f"{dead} is installed but never imported"
        )


def test_every_requirement_is_importable_by_some_module(requirements):
    """A dependency nothing imports is dead weight; keep the list honest."""
    import re as _re

    sources = []
    for root, _dirs, files in os.walk(os.path.join(REPO_ROOT, "scripts")):
        if "__pycache__" in root:
            continue
        for name in files:
            if name.endswith(".py"):
                with open(os.path.join(root, name)) as f:
                    sources.append(f.read())
    blob = "\n".join(sources)

    module_for = {
        "torch": "torch",
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
        "pyyaml": "yaml",
        "tqdm": "tqdm",
        "torchvision": "torchvision",
        "ultralytics": "ultralytics",
        "opencv-python": "cv2",
        "kagglehub": "kagglehub",
    }
    for requirement in requirements:
        package = _re.split(r"[=<>!~]", requirement)[0].strip()
        module = module_for.get(package)
        if module is None:
            continue
        assert _re.search(rf"\b{_re.escape(module)}\b", blob), f"{package} is declared but never imported"


def test_python_version_guard():
    assert check_python_version() is True
    with pytest.raises(SystemExit, match="Python"):
        check_python_version((99, 0))


def test_supported_python_matches_pyproject():
    import sys

    assert MIN_PYTHON == (3, 9)
    with open(os.path.join(REPO_ROOT, "pyproject.toml")) as f:
        pyproject = f.read()
    assert 'requires-python = ">=3.9"' in pyproject, "pyproject and setup.py must agree"
    assert sys.version_info[:2] >= MIN_PYTHON, "tests are running on an unsupported interpreter"


def test_config_is_the_single_source_of_hyperparameters():
    from scripts.config import CONFIG

    for key in ["random_seed", "val_split", "normal_action_ids", "feature_layer", "seq_len"]:
        assert key in CONFIG, f"{key} is read by the pipeline but missing from CONFIG"
