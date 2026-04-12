#!/usr/bin/env python3
"""Export YOLOE ONNX with class prompts baked into the model metadata."""

from __future__ import annotations

from pathlib import Path
import re

from ultralytics import YOLO

# Edit these values directly.
MODEL_PATH = Path("/home/ferdinand/unitree/go2_fetch_ros2/fetch/models/yoloe-26l-seg.pt")
CLASSES = ["ball", "round object", "sphere", "round"]
IMGSZ = 640 # Inference resolution (square input size used in ONNX export graph).
OPSET = 17 # ONNX operator set version; keep in sync with your runtime compatibility.
DEVICE = "0"  # Examples: "0", "cpu"
SIMPLIFY = False


def _slugify(token: str) -> str:
    normalized = re.sub(r"\s+", "_", token.strip().lower())
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def main() -> None:
    model_path = MODEL_PATH.expanduser().resolve()

    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Expected a .pt model for prompt setup, got: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    classes = [c.strip() for c in CLASSES if c.strip()]
    if not classes:
        raise ValueError("At least one non-empty class prompt is required.")

    print(f"Applying class prompts: {classes}")
    model.set_classes(classes)

    print("Exporting ONNX...")
    exported_path = model.export(
        format="onnx",
        imgsz=IMGSZ,
        opset=OPSET,
        device=DEVICE,
        simplify=SIMPLIFY,
    )
    exported_path = Path(str(exported_path))

    class_suffix = "_".join(_slugify(c) for c in classes if _slugify(c))
    base_name = model_path.stem
    target_stem = f"{base_name}_{class_suffix}" if class_suffix else base_name
    target_path = exported_path.with_name(f"{target_stem}{exported_path.suffix}")

    if target_path != exported_path:
        if target_path.exists():
            target_path.unlink()
        exported_path.rename(target_path)
        exported_path = target_path

    print(f"Export complete: {exported_path}")


if __name__ == "__main__":
    main()
