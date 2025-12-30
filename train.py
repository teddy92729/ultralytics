from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/rgbt/yolov8.yaml")

results = model.train(
    data="/opt/datasets/DroneVehicle/data.yaml",
    epochs=150,
    imgsz=640,
    device=0,
    batch=8,
    rgbt=True,
    name="DV_YOLOv8s",
)
