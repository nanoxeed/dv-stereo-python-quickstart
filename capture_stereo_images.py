from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2 as cv
import dv_processing as dv

from stereo_common import (
    add_accumulator_args,
    add_camera_selection_args,
    add_slicer_args,
    cameras_are_running,
    create_stereo_accumulators,
    frame_image,
    next_stereo_events,
    open_stereo_cameras,
    side_by_side,
    slicer_interval,
    stereo_geometry_from_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture accumulated stereo images from synchronized event cameras.")
    add_camera_selection_args(parser)
    add_slicer_args(parser)
    add_accumulator_args(parser)
    parser.add_argument("--output-dir", type=Path, default=Path("captures"))
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--auto-interval-sec", type=float, default=0.0)
    parser.add_argument("--calibration", type=Path, help="Optional calibration JSON for rectified image output.")
    parser.add_argument("--left-designation", default="C0")
    parser.add_argument("--right-designation", default="C1")
    parser.add_argument("--window-name", default="Stereo capture")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stereo_geometry = None
    calibration = None
    if args.calibration is not None:
        calibration, stereo_geometry = stereo_geometry_from_file(
            args.calibration, args.left_designation, args.right_designation
        )

    if calibration is not None:
        # Open by the calibration's left/right names so the LEFT/RIGHT rectification maps are
        # applied to the matching cameras even if USB enumeration order differs from
        # calibration time (otherwise the *_rectified.png pair can come out swapped).
        pair = open_stereo_cameras(
            args.left or calibration.left.name,
            args.right or calibration.right.name,
        )
    else:
        pair = open_stereo_cameras(args.left, args.right)
    left_accumulator, right_accumulator = create_stereo_accumulators(pair, args)

    latest_left = None
    latest_right = None
    saved_count = 0
    last_auto_save = 0.0
    keep_running = True

    def save_latest() -> None:
        nonlocal saved_count
        if latest_left is None or latest_right is None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sample_dir = args.output_dir / timestamp
        sample_dir.mkdir(parents=True, exist_ok=False)

        cv.imwrite(str(sample_dir / "left.png"), latest_left)
        cv.imwrite(str(sample_dir / "right.png"), latest_right)

        metadata = {
            "timestamp": timestamp,
            "left_camera": pair.left_name,
            "right_camera": pair.right_name,
            "accumulator": args.accumulator,
            "interval_ms": args.interval_ms,
        }

        if stereo_geometry is not None:
            left_rectified = stereo_geometry.remapImage(dv.camera.StereoGeometry.CameraPosition.LEFT, latest_left)
            right_rectified = stereo_geometry.remapImage(dv.camera.StereoGeometry.CameraPosition.RIGHT, latest_right)
            cv.imwrite(str(sample_dir / "left_rectified.png"), left_rectified)
            cv.imwrite(str(sample_dir / "right_rectified.png"), right_rectified)
            metadata["calibration"] = str(args.calibration)

        (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        saved_count += 1
        print(f"Saved {sample_dir}")

    cv.namedWindow(args.window_name, cv.WINDOW_NORMAL)
    slicer = dv.StereoEventStreamSlicer()

    def preview(left_events, right_events) -> None:
        nonlocal latest_left, latest_right, last_auto_save, keep_running
        left_accumulator.accept(left_events)
        right_accumulator.accept(right_events)
        latest_left = frame_image(left_accumulator.generateFrame()).copy()
        latest_right = frame_image(right_accumulator.generateFrame()).copy()
        cv.imshow(args.window_name, side_by_side(latest_left, latest_right, pair.left_name, pair.right_name))

        now = time.monotonic()
        if args.auto_interval_sec > 0.0 and now - last_auto_save >= args.auto_interval_sec:
            save_latest()
            last_auto_save = now
            if saved_count >= args.count:
                keep_running = False

    slicer.doEveryTimeInterval(slicer_interval(args), preview)

    while keep_running and cameras_are_running(pair):
        slicer.accept(*next_stereo_events(pair))
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            keep_running = False
        elif key in (ord(" "), ord("s")):
            save_latest()
            if saved_count >= args.count:
                keep_running = False

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
