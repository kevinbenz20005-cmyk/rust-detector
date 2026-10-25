
from ultralytics import YOLO


def main():
    # Convert the trained model to ONNX so the website can run it in the browser.
    # Output shape is (1, 5, 8400): cx, cy, w, h, score. The page runs NMS itself,
    # which matches how Ultralytics predicts with this model.
    model = YOLO("runs/detect/runs/corrosion/yolov10s/weights/best.pt")
    path = model.export(format="onnx", imgsz=640, opset=17, simplify=True)

    import shutil
    shutil.copy(path, "public/model/best.onnx")
    print("Copied to public/model/best.onnx")


if __name__ == "__main__":
    main()
