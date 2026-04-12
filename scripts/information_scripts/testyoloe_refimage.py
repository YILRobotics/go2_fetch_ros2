import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# 1. Initialize the model
model = YOLOE("yoloe-26l-seg.pt")

# 2. Define the cube's location in your REFERENCE image
# Note: These coordinates MUST match where the cube is in 'cube_ref.jpg'
visual_prompts = dict(
    bboxes=np.array([[280, 295, 535, 1112]]), # [x1, y1, x2, y2]
    cls=np.array([0]), # Class ID (e.g., 0 for 'target_cube')
)

# 3. Run prediction on the webcam (source=0)
# We use an explicit VideoCapture loop so we can request camera-side FPS.
camera_index = 1
target_hz = 20
window_name = "YOLOE Visual Prompting"

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {camera_index}")

cap.set(cv2.CAP_PROP_FPS, target_hz)
reported_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Requested camera FPS: {target_hz} | Camera-reported FPS: {reported_fps:.2f}")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Could not read frame from camera.")
        break

    results = model.predict(
        source=frame,
        refer_image="D:\\programming\\Spot-Light\\bottle_reference.jpg",
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        conf=0.04,
        verbose=False,
    )

    # If you need to send coordinates to your Franka:
    # boxes = results[0].boxes.xyxy
    # masks = results[0].masks.data if results[0].masks is not None else None

    annotated = results[0].plot()
    cv2.imshow(window_name, annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

