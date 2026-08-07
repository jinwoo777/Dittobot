#!/usr/bin/env python3
"""Minimal Intel RealSense D435i stream diagnostic for Ubuntu/Linux."""
from __future__ import annotations

import argparse
import time
from importlib.metadata import PackageNotFoundError, version


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def info_or_unknown(device, rs, field) -> str:
    try:
        return device.get_info(field)
    except Exception:
        return "unknown"


def safe_stop(pipeline) -> None:
    try:
        pipeline.stop()
    except RuntimeError:
        pass


def test_stream(rs, serial: str | None, mode: str, width: int, height: int, fps: int, frames: int) -> bool:
    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    if mode in {"depth", "both"}:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    if mode in {"color", "both"}:
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    started = False
    try:
        print(f"\n[TEST] {mode}: {width}x{height}@{fps}")
        profile = pipeline.start(config)
        started = True
        device = profile.get_device()
        print("  device:", info_or_unknown(device, rs, rs.camera_info.name))
        print("  serial:", info_or_unknown(device, rs, rs.camera_info.serial_number))
        for i in range(frames):
            fs = pipeline.wait_for_frames(timeout_ms=5000)
            if i in {0, frames - 1}:
                print(f"  frame {i + 1}/{frames}: OK")
        print(f"[PASS] {mode}")
        return True
    except RuntimeError as exc:
        print(f"[FAIL] {mode}: {exc}")
        return False
    finally:
        if started:
            safe_stop(pipeline)
        time.sleep(0.5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        print("pyrealsense2 import failed:", exc)
        return 2

    print("pyrealsense2 package:", package_version("pyrealsense2"))
    ctx = rs.context()
    devices = list(ctx.query_devices())
    print("RealSense device count:", len(devices))
    if not devices:
        print("No RealSense device detected.")
        return 3

    for index, device in enumerate(devices):
        print(f"\n[DEVICE {index}]")
        print("  name:", info_or_unknown(device, rs, rs.camera_info.name))
        print("  serial:", info_or_unknown(device, rs, rs.camera_info.serial_number))
        print("  firmware:", info_or_unknown(device, rs, rs.camera_info.firmware_version))
        if hasattr(rs.camera_info, "usb_type_descriptor"):
            print("  usb:", info_or_unknown(device, rs, rs.camera_info.usb_type_descriptor))

    serial = args.serial
    if serial is None and len(devices) == 1:
        serial = info_or_unknown(devices[0], rs, rs.camera_info.serial_number)

    results = {
        mode: test_stream(rs, serial, mode, args.width, args.height, args.fps, args.frames)
        for mode in ("depth", "color", "both")
    }
    print("\n=== SUMMARY ===")
    for mode, ok in results.items():
        print(f"{mode:5s}: {'PASS' if ok else 'FAIL'}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
