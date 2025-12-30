from ultralytics import YOLO

model = YOLO("runs/obb/DV_YOLOv8s/weights/best.pt")

results = model.val(
    data="/opt/datasets/DroneVehicle/data.yaml",
    imgsz=640,
    device=0,
    batch=8,
    rgbt=True,
)