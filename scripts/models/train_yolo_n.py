"""Train YOLO11n on cow detection dataset (from notebook 1)."""

from ultralytics import YOLO


def train_yolo11n(data_yaml: str = "/content/cow_detection.yaml"):
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=data_yaml,
        epochs=100,
        batch=16,
        imgsz=640,
        cache=False,
        workers=2,
    )
    return results


if __name__ == "__main__":
    train_yolo11n()
