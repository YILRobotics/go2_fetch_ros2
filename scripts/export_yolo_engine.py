
from ultralytics import YOLO

# 1. Load the open-vocabulary segmentation model
model = YOLO("/home/unitree/fetch_ws/src/go2_fetch_ros2/fetch/models/yolo/yoloe-26s-seg.pt")
model.set_classes(["box"])

# 2. Export to TensorRT (.engine) with half-precision (FP16)
# This will handle the ONNX middleman conversion automatically
model.export(
    format="engine",
    imgsz=(360, 640), # because model needs image dimensions divisible by the model stride, typically 32. Prevents padding.
    half=True,
    dynamic=False,
    batch=1,
    device=0,
)