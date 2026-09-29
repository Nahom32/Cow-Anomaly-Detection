"""Install the project's dependencies.

This used to carry its own hardcoded package list, which had already drifted
from `requirements.txt` (it omitted matplotlib, pyyaml, numpy and scikit-learn,
and pinned nothing). It now installs that file and nothing else — one list, one
place to update.
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REQUIREMENTS = os.path.join(REPO_ROOT, "requirements.txt")

MIN_PYTHON = (3, 9)


def check_python_version(minimum=MIN_PYTHON):
    """Fail early rather than with a TypeError from `str | None` deep in a run.

    The README used to claim 3.10+ while the code used PEP-604 annotations, so a
    3.9 machine got a confusing error partway through the pipeline.
    """
    if sys.version_info < minimum:
        required = ".".join(str(part) for part in minimum)
        current = ".".join(str(part) for part in sys.version_info[:3])
        raise SystemExit(
            f"Python {required}+ is required (running {current}). "
            f"Create a virtualenv with a supported interpreter and re-run."
        )
    return True


def read_requirements(path=DEFAULT_REQUIREMENTS):
    """Parse a requirements file into the list of package specifiers."""
    requirements = []
    with open(path) as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            requirements.append(line)
    return requirements


def install_dependencies(requirements_path=DEFAULT_REQUIREMENTS, dry_run=False):
    """Install every requirement via a single `pip install -r` call."""
    requirements = read_requirements(requirements_path)
    if not requirements:
        raise SystemExit(f"no requirements found in {requirements_path}")

    command = [sys.executable, "-m", "pip", "install", "-r", requirements_path]
    print(f"Installing {len(requirements)} requirement(s) from {requirements_path}")
    for req in requirements:
        print(f"  {req}")
    if dry_run:
        print(" ".join(command))
        return requirements
    subprocess.check_call(command)
    return requirements


def main():
    check_python_version()
    parser = argparse.ArgumentParser(description="Install Cow-Anomaly-Detection dependencies")
    parser.add_argument("--requirements", default=DEFAULT_REQUIREMENTS,
                        help="Path to a requirements file (default: the repo's requirements.txt)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be installed and exit")
    args = parser.parse_args()
    install_dependencies(args.requirements, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
