
from ultralytics import YOLO


def main():
    model = YOLO("runs/detect/runs/corrosion/yolov10s/weights/best.pt")

    metrics = model.val(
        data="data.yaml",
        split="test",
        workers=0,
        device=0
    )

    print("\nTest evaluation completed!")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
