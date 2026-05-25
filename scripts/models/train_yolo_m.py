from ultralytics import YOLO


def train_yolo26m(data_yaml: str = "/content/cow_detection_fixed.yaml"):
    model = YOLO("yolo26m.pt")
    results = model.train(
        data=data_yaml,
        epochs=150,
        imgsz=640,
        batch=16,
        patience=50,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.2,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        optimizer="auto",
        device=0,
        workers=4,
        project="cow_detector",
        name="yolo26m_cbvd",
        exist_ok=True,
        verbose=True,
    )
    return results


if __name__ == "__main__":
    train_yolo26m()
