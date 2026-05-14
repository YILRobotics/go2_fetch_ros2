
from ultralytics import YOLO
import cv2
import time

cap = cv2.VideoCapture(4) # 4 is colour of realsense d435i, 0 is depth of realsense d435i

# model = YOLO("yolov8m-world.pt")
model = YOLO("/home/ferdinand/unitree/go2_fetch_ros2/fetch/models/yoloe-26l-seg.pt")
# model = YOLO("yolov8l-worldv2.pt")
# model.set_classes(["black box", "box", "cube", "black cube"])
# model.set_classes(["tube", "cylinder", "can", "bottle"])
model.set_classes(["ball"])

last_fps_time = time.perf_counter()
frame_counter = 0
last_no_det_print_time = time.perf_counter()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model.predict(frame, verbose=False, conf=0.15)
    annotated = results[0].plot()

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        labels = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            labels.append(f"{model.names[cls_id]}:{conf:.2f}")
        print(f"Detections ({len(labels)}): {', '.join(labels)}")
    else:
        now_no_det = time.perf_counter()
        if (now_no_det - last_no_det_print_time) >= 1.0:
            print("No detections")
            last_no_det_print_time = now_no_det

    frame_counter += 1
    now = time.perf_counter()
    elapsed = now - last_fps_time
    if elapsed >= 1.0:
        fps = frame_counter / elapsed
        print(f"Loop frequency: {fps:.2f} Hz")
        frame_counter = 0
        last_fps_time = now

    cv2.imshow("Open-vocabulary detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

#########################################

# while True:
#     ok, frame = cap.read()
#     if not ok:
#         print("Could not read from camera.")
#         break

#     results = model.predict(frame, conf=0.15, imgsz=640, device=0, verbose=False)
#     annotated = results[0].plot()

#     cv2.imshow("YOLOE-26L", annotated)

#     if results[0].boxes is not None and len(results[0].boxes) > 0:
#         print("Detections:")
#         for box in results[0].boxes:
#             cls_id = int(box.cls[0])
#             conf = float(box.conf[0])
#             print(f"  {model.names[cls_id]}: {conf:.2f}")

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

