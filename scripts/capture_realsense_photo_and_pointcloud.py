#!/usr/bin/env python3
"""Capture one RealSense RGB frame and save image + colorized point cloud artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit(
        "pyrealsense2 is required. Install Intel RealSense SDK Python bindings."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one frame from RealSense and save image + point cloud."
    )
    parser.add_argument(
        "--output-dir",
        default="/home/ferdinand/unitree/go2_fetch_ros2/data",
        help="Directory where artifacts are written.",
    )
    parser.add_argument(
        "--image-name",
        default="realsense_color.png",
        help="Output image filename.",
    )
    parser.add_argument(
        "--cloud-ply-name",
        default="realsense_points.ply",
        help="Output point cloud PLY filename.",
    )
    parser.add_argument(
        "--cloud-npy-name",
        default="realsense_points.npy",
        help="Output point cloud NPY filename (Nx4 float32: x, y, z, packed_rgb).",
    )
    parser.add_argument("--width", type=int, default=640, help="Stream width.")
    parser.add_argument("--height", type=int, default=480, help="Stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Stream FPS.")
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=60,
        help="Frames to drop before capture.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=5000,
        help="Wait timeout for each frame.",
    )
    return parser.parse_args()


def _pack_rgb_float(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    rgb_uint32 = (
        (red.astype(np.uint32) << 16)
        | (green.astype(np.uint32) << 8)
        | blue.astype(np.uint32)
    )
    return rgb_uint32.view(np.float32)


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / args.image_name
    cloud_ply_path = output_dir / args.cloud_ply_name
    cloud_npy_path = output_dir / args.cloud_npy_name

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)
    pointcloud = rs.pointcloud()

    try:
        pipeline.start(config)
    except Exception as exc:
        print(f"Failed to start RealSense pipeline: {exc}", file=sys.stderr)
        return 1

    try:
        for _ in range(max(args.warmup_frames, 0)):
            pipeline.wait_for_frames(timeout_ms=args.timeout_ms)

        frames = pipeline.wait_for_frames(timeout_ms=args.timeout_ms)
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("Failed to get valid color/depth frames.", file=sys.stderr)
            return 1

        color_image = np.asanyarray(color_frame.get_data())
        if not cv2.imwrite(str(image_path), color_image):
            print(f"Failed to save image to {image_path}", file=sys.stderr)
            return 1

        pointcloud.map_to(color_frame)
        points = pointcloud.calculate(depth_frame)
        points.export_to_ply(str(cloud_ply_path), color_frame)

        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        color_pixels = color_image.reshape(-1, 3)
        point_count = min(vertices.shape[0], color_pixels.shape[0])
        vertices = vertices[:point_count]
        color_pixels = color_pixels[:point_count]

        valid = np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 0.0)
        vertices = vertices[valid]
        color_pixels = color_pixels[valid]

        rgb_float = _pack_rgb_float(color_pixels[:, 2], color_pixels[:, 1], color_pixels[:, 0])
        cloud_rgba = np.column_stack((vertices, rgb_float.astype(np.float32, copy=False)))
        np.save(cloud_npy_path, cloud_rgba.astype(np.float32, copy=False))

        print(f"Saved image: {image_path}")
        print(f"Saved point cloud (PLY): {cloud_ply_path}")
        print(f"Saved point cloud (NPY): {cloud_npy_path} ({cloud_rgba.shape[0]} points, xyzrgb)")
        return 0
    except Exception as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        return 1
    finally:
        pipeline.stop()
        time.sleep(0.1)


if __name__ == "__main__":
    raise SystemExit(main())
