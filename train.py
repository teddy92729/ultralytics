from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/rgbt/warp_net_wo_aefm.yaml")

results = model.train(
    data="/opt/datasets/DroneVehicle/data.yaml",
    epochs=100,
    imgsz=640,
    device=0,
    batch=8,
    rgbt=True,
    augment=False,
    name="DV_WarpNet_wo_aefm",
)
