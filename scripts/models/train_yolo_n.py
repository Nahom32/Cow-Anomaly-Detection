"""YOLO detector training.

Every hyperparameter comes from CONFIG["yolo"] — hardcoding them here is what
made the recorded settings disagree with the settings that were run (2.12).
"""

from scripts.config import CONFIG
from scripts.utils.seeding import set_seed


def train_yolo(data_yaml, model="yolo26n.pt", seed=None, overrides=None, project=None, name=None):
    from ultralytics import YOLO

    if seed is None:
        seed = CONFIG["random_seed"]
    set_seed(seed)

    settings = dict(CONFIG["yolo"])
    settings.update(overrides or {})
    if project is not None:
        settings["project"] = project
    if name is not None:
        settings["name"] = name

    results = YOLO(model).train(data=data_yaml, seed=seed, **settings)
    return results


def train_yolo26n(data_yaml, seed=None, **kwargs):
    return train_yolo(data_yaml, model=CONFIG["yolo"]["model"], seed=seed, **kwargs)


def train_yolo26m(data_yaml, seed=None, **kwargs):
    return train_yolo(data_yaml, model="yolo26m.pt", seed=seed, **kwargs)


if __name__ == "__main__":
    train_yolo26n(CONFIG.get("yolo_data_yaml", "/content/cow_detection_fixed.yaml"))
