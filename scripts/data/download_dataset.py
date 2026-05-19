import os

import kagglehub


def download_dataset() -> str:
    path = kagglehub.dataset_download(
        "fandaoerji/cbvd-5cow-behavior-video-dataset"
    )
    print(f"Dataset downloaded to: {path}")
    for d in os.listdir(path):
        print(f"  {d}")
    return path


if __name__ == "__main__":
    download_dataset()
