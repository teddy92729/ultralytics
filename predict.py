from ultralytics import YOLO
import cv2

model = YOLO("runs/obb/DV_YOLOv8s/weights/best.pt")

img = cv2.imread("/opt/datasets/DroneVehicle/test/visible/images/00014.jpg", cv2.IMREAD_COLOR)
img_t = cv2.imread("/opt/datasets/DroneVehicle/test/infrared/images/00014.jpg", cv2.IMREAD_GRAYSCALE)
b, g, r = cv2.split(img)
rgbt_img = cv2.merge([r, g, b, img_t])

results = model(rgbt_img, save=True, save_txt=True, visualize=True)