import os


def download_dataset() -> str:
    # Imported here so that modules which only *accept* a data root (and every
    # test) do not need the Kaggle stack installed just to be importable.
    import kagglehub

    path = kagglehub.dataset_download(
        "fandaoerji/cbvd-5cow-behavior-video-dataset"
    )
    print(f"Dataset downloaded to: {path}")
    for d in os.listdir(path):
        print(f"  {d}")
    return path


if __name__ == "__main__":
    download_dataset()
