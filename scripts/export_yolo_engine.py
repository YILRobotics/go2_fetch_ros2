
from ultralytics import YOLO

# 1. Load the open-vocabulary segmentation model
model = YOLO("/home/unitree/fetch_ws/src/go2_fetch_ros2/fetch/models/yolo/yoloe-26n-seg.pt")
model.set_classes(["box"])

# 2. Export to TensorRT (.engine) with half-precision (FP16)
# This will handle the ONNX middleman conversion automatically
model.export(format="engine", half=True, device=0)
