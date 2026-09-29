"""YOLO26m training.

The training settings live in CONFIG["yolo"] and are shared with the 26n
trainer; the only difference is the model size, so this is a thin wrapper rather
than a second copy of two dozen hyperparameters.
"""

from scripts.config import CONFIG
from scripts.models.train_yolo_n import train_yolo


def train_yolo26m(data_yaml, seed=None, **kwargs):
    return train_yolo(
        data_yaml,
        model="yolo26m.pt",
        seed=seed,
        overrides={"name": "yolo26m_cbvd"},
        **kwargs,
    )


if __name__ == "__main__":
    train_yolo26m(CONFIG.get("yolo_data_yaml", "/content/cow_detection_fixed.yaml"))
