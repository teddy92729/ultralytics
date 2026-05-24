from ultralytics import YOLO, RTDETR

model = YOLO("runs/obb/DV_WarpNet_wo_aefm/weights/best.pt")
# model = RTDETR("runs/detect/DV_RT-DETR_resnet50/weights/best.pt")

results = model.val(
    # data="/opt/datasets/DroneVehicle/VEDAI/YOLO1024/data.yaml",
    split="test",
    data="/opt/datasets/DroneVehicle/data.yaml",
    imgsz=640,
    device=0,
    batch=8,
    rgbt=True,
)