
from ultralytics import YOLO

# Load pretrained YOLOv10 Small
model = YOLO("yolov10s.pt")

# Train on your corrosion dataset
results = model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    batch=4,
    workers=0,
    device=0,
    project="runs/corrosion",
    name="yolov10s",
    plots=True
)
