import subprocess
import sys


def install_dependencies():
    packages = [
        "ultralytics",
        "scipy",
        "torch",
        "torchvision",
        "pytorchvideo",
        "decord",
        "opencv-python",
        "tqdm",
        "pandas",
        "kagglehub",
    ]
    for pkg in packages:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg]
        )


if __name__ == "__main__":
    install_dependencies()
