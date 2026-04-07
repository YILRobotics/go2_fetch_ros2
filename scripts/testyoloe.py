
from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(1)

# model = YOLO("yolov8m-world.pt")
model = YOLO("yoloe-26l-seg.pt")
# model = YOLO("yolov8l-worldv2.pt")
# model.set_classes(["black box", "box", "cube", "black cube"])
# model.set_classes(["tube", "cylinder", "can", "bottle"])
model.set_classes(["bottle"])

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model.predict(frame, verbose=False, conf=0.15)
    annotated = results[0].plot()

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

