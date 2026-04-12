from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    xyxy: np.ndarray
    conf: float
    cls_id: int
    cls_name: str


class InteractiveBBoxFinder:
    WINDOW_NAME = "Interactive Bounding Box Finder"

    def __init__(self, image: np.ndarray, detections: list[Detection], start_conf: float) -> None:
        self.image = image
        self.detections = detections
        self.conf_threshold = float(np.clip(start_conf, 0.0, 1.0))
        self.selected_idx: Optional[int] = None

    def _visible_indices(self) -> list[int]:
        return [index for index, det in enumerate(self.detections) if det.conf >= self.conf_threshold]

    def _ensure_selected_visible(self) -> None:
        visible = self._visible_indices()
        if not visible:
            self.selected_idx = None
            return
        if self.selected_idx not in visible:
            self.selected_idx = visible[0]

    def _selected_detection(self) -> Optional[Detection]:
        if self.selected_idx is None:
            return None
        return self.detections[self.selected_idx]

    def _print_selected(self) -> None:
        det = self._selected_detection()
        if det is None:
            print("No bounding box selected.")
            return

        x1, y1, x2, y2 = [int(round(value)) for value in det.xyxy.tolist()]
        print(
            f"Selected bbox: [{x1}, {y1}, {x2}, {y2}] "
            f"| class={det.cls_name} (id={det.cls_id}) | conf={det.conf:.3f}"
        )

    def _on_trackbar(self, value: int) -> None:
        self.conf_threshold = value / 100.0
        self._ensure_selected_visible()

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: int) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        best_idx: Optional[int] = None
        best_area: Optional[float] = None

        for idx in self._visible_indices():
            det = self.detections[idx]
            x1, y1, x2, y2 = det.xyxy.tolist()
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = max(1.0, (x2 - x1) * (y2 - y1))
                if best_area is None or area < best_area:
                    best_area = area
                    best_idx = idx

        self.selected_idx = best_idx
        self._print_selected()

    def _cycle_selection(self, direction: int) -> None:
        visible = self._visible_indices()
        if not visible:
            self.selected_idx = None
            return

        if self.selected_idx not in visible:
            self.selected_idx = visible[0]
        else:
            pos = visible.index(self.selected_idx)
            self.selected_idx = visible[(pos + direction) % len(visible)]

        self._print_selected()

    def _draw_label(self, frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        (width, height), _ = cv2.getTextSize(text, font, scale, thickness)
        y = max(height + 6, y)

        cv2.rectangle(frame, (x, y - height - 6), (x + width + 6, y), color, -1)
        cv2.putText(frame, text, (x + 3, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _render(self) -> np.ndarray:
        frame = self.image.copy()
        h, w = frame.shape[:2]
        visible = self._visible_indices()

        for idx in visible:
            det = self.detections[idx]
            x1, y1, x2, y2 = [int(round(value)) for value in det.xyxy.tolist()]
            x1 = int(np.clip(x1, 0, max(0, w - 1)))
            x2 = int(np.clip(x2, 0, max(0, w - 1)))
            y1 = int(np.clip(y1, 0, max(0, h - 1)))
            y2 = int(np.clip(y2, 0, max(0, h - 1)))

            is_selected = idx == self.selected_idx
            color = (0, 0, 255) if is_selected else (0, 180, 0)
            thickness = 3 if is_selected else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"{det.cls_name} {det.conf:.2f}"
            self._draw_label(frame, label, x1, y1 - 4, color)

        info_text = (
            f"conf>={self.conf_threshold:.2f}  visible:{len(visible)}/{len(self.detections)}  "
            "[click] select  [n/p] next/prev  [+/-] conf  [enter] print  [q/esc] quit"
        )

        cv2.rectangle(frame, (0, 0), (w, 30), (20, 20, 20), -1)
        cv2.putText(frame, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

        det = self._selected_detection()
        if det is not None:
            x1, y1, x2, y2 = [int(round(value)) for value in det.xyxy.tolist()]
            selected_text = (
                f"Selected: [{x1}, {y1}, {x2}, {y2}]  "
                f"class={det.cls_name}  conf={det.conf:.3f}"
            )
            cv2.rectangle(frame, (0, h - 28), (w, h), (20, 20, 20), -1)
            cv2.putText(frame, selected_text, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

        return frame

    def run(self) -> None:
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)
        cv2.createTrackbar(
            "Confidence %",
            self.WINDOW_NAME,
            int(self.conf_threshold * 100),
            100,
            self._on_trackbar,
        )

        self._ensure_selected_visible()
        if self.selected_idx is not None:
            self._print_selected()

        while True:
            frame = self._render()
            cv2.imshow(self.WINDOW_NAME, frame)

            key = cv2.waitKey(16) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("n"):
                self._cycle_selection(+1)
            elif key == ord("p"):
                self._cycle_selection(-1)
            elif key in (ord("+"), ord("=")):
                pos = cv2.getTrackbarPos("Confidence %", self.WINDOW_NAME)
                cv2.setTrackbarPos("Confidence %", self.WINDOW_NAME, min(100, pos + 1))
            elif key in (ord("-"), ord("_")):
                pos = cv2.getTrackbarPos("Confidence %", self.WINDOW_NAME)
                cv2.setTrackbarPos("Confidence %", self.WINDOW_NAME, max(0, pos - 1))
            elif key == 13:
                self._print_selected()

        cv2.destroyAllWindows()


def _resolve_existing_path(path_text: str, script_dir: Path) -> Path:
    candidate = Path(path_text)
    if candidate.exists():
        return candidate

    candidate_from_script = script_dir / candidate
    if candidate_from_script.exists():
        return candidate_from_script

    return candidate


def _parse_classes(classes_arg: str) -> list[str]:
    return [item.strip() for item in classes_arg.split(",") if item.strip()]


def _class_name_for_id(names, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def _detect_once(
    image_path: Path,
    model_path: Path,
    classes: list[str],
    scan_conf: float,
    start_conf: float,
    imgsz: int,
    iou: float,
    device: Optional[str],
) -> tuple[np.ndarray, list[Detection]]:
    model = YOLO(str(model_path))

    if classes:
        try:
            model.set_classes(classes)
            print(f"Using open-vocabulary classes: {classes}")
        except Exception as exc:
            print(f"Warning: could not set classes on this model ({exc}). Continuing with default classes.")

    results = model.predict(
        source=str(image_path),
        conf=float(np.clip(scan_conf, 0.0, 1.0)),
        iou=float(np.clip(iou, 0.0, 1.0)),
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    result = results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return image, []

    names = result.names if hasattr(result, "names") else getattr(model, "names", {})
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    detections = [
        Detection(
            xyxy=xyxy[i],
            conf=float(conf[i]),
            cls_id=int(cls_ids[i]),
            cls_name=_class_name_for_id(names, int(cls_ids[i])),
        )
        for i in range(len(xyxy))
    ]
    detections.sort(key=lambda det: det.conf, reverse=True)

    if detections and all(det.conf < start_conf for det in detections):
        max_conf = max(det.conf for det in detections)
        print(
            f"Note: no detections above start threshold ({start_conf:.2f}). "
            f"Highest detection confidence is {max_conf:.2f}."
        )

    return image, detections


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fast interactive tool to find bounding boxes for objects in a single image."
    )
    parser.add_argument("--image", "-i", default="D:\\programming\\Spot-Light\\bottle_reference.jpg", help="Path to input image")
    parser.add_argument(
        "--model",
        "-m",
        default="yoloe-26l-seg.pt",
        help="Path to YOLO/YOLOE/YOLO-world model",
    )
    parser.add_argument(
        "--classes",
        "-c",
        default="bottle",
        help="Comma-separated classes for open-vocabulary models (example: 'box,cube,bottle')",
    )
    parser.add_argument(
        "--scan-conf",
        type=float,
        default=0.01,
        help="Confidence used for initial detection pass (lower keeps more candidates)",
    )
    parser.add_argument(
        "--start-conf",
        type=float,
        default=0.25,
        help="Initial confidence threshold in interactive view",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device (e.g. '0' for first GPU, 'cpu' for CPU). Defaults to Ultralytics auto.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    script_dir = Path(__file__).resolve().parent

    image_path = _resolve_existing_path(args.image, script_dir)
    model_path = _resolve_existing_path(args.model, script_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model does not exist: {model_path}")

    classes = _parse_classes(args.classes)

    image, detections = _detect_once(
        image_path=image_path,
        model_path=model_path,
        classes=classes,
        scan_conf=args.scan_conf,
        start_conf=args.start_conf,
        imgsz=args.imgsz,
        iou=args.iou,
        device=args.device,
    )

    print(f"Loaded image: {image_path}")
    print(f"Loaded model: {model_path}")
    print(f"Found {len(detections)} raw detections.")

    viewer = InteractiveBBoxFinder(image=image, detections=detections, start_conf=args.start_conf)
    viewer.run()


if __name__ == "__main__":
    main()
