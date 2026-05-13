from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2 as cv
import dv_processing as dv
import numpy as np

from stereo_common import (
    add_camera_selection_args,
    add_slicer_args,
    cameras_are_running,
    load_stereo_calibration,
    next_stereo_events,
    open_stereo_cameras,
    slicer_interval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate stereo depth from events and visualize the point cloud in Rerun.")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--left-designation", default="C0")
    parser.add_argument("--right-designation", default="C1")
    add_camera_selection_args(parser)
    add_slicer_args(parser, default_interval_ms=100)
    parser.add_argument("--event-count", type=int, help="Overlapping event buffer size per camera.")
    parser.add_argument("--max-points", type=int, default=40_000)
    parser.add_argument("--min-disparity", type=float, default=1.0)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--point-radius", type=float, default=0.003)
    parser.add_argument("--median-filter-size", type=int, default=3, help="Odd kernel size for disparity median filtering. Use 0 to disable.")
    parser.add_argument("--speckle-size", type=int, default=100, help="Remove disparity blobs up to this pixel size. Use 0 to disable.")
    parser.add_argument("--speckle-diff", type=float, default=2.0, help="Maximum disparity difference in px for speckle filtering.")
    parser.add_argument("--matcher-contribution", type=float, default=0.25, help="Event contribution for the matcher EdgeMapAccumulator.")
    parser.add_argument("--matcher-decay", type=float, default=1.0, help="Decay for the matcher EdgeMapAccumulator.")
    parser.add_argument("--use-polarity", action="store_false", dest="matcher_ignore_polarity", help="Keep event polarity in matcher accumulation.")
    parser.add_argument("--ignore-polarity", action="store_true", dest="matcher_ignore_polarity", help="Ignore event polarity in matcher accumulation.")
    parser.add_argument("--pointcloud-interval", type=int, default=2, help="Log point clouds to Rerun every N frames.")
    parser.add_argument("--image-interval", type=int, default=2, help="Log stereo/disparity images to Rerun every N frames.")
    parser.add_argument("--stats-interval", type=int, default=30, help="Print disparity/point-cloud stats every N frames.")
    parser.add_argument("--rerun-memory-limit", default="50%", help="Viewer memory limit used when spawning Rerun.")
    parser.add_argument("--rerun-server-memory-limit", default="512MiB", help="Rerun server memory limit used when spawning Rerun.")
    parser.add_argument("--connect", action="store_true", help="Connect to an already running Rerun viewer.")
    parser.add_argument("--save-rrd", type=Path, help="Write a Rerun recording instead of spawning a viewer.")
    parser.set_defaults(matcher_ignore_polarity=True)
    args = parser.parse_args()
    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")
    if args.event_count is not None and args.event_count <= 0:
        parser.error("--event-count must be positive")
    if args.max_points < 0:
        parser.error("--max-points must be zero or positive")
    if args.pointcloud_interval < 0:
        parser.error("--pointcloud-interval must be zero or positive")
    if args.image_interval < 0:
        parser.error("--image-interval must be zero or positive")
    if args.median_filter_size < 0:
        parser.error("--median-filter-size must be zero or positive")
    if args.median_filter_size > 0 and args.median_filter_size % 2 == 0:
        parser.error("--median-filter-size must be odd")
    if args.speckle_size < 0:
        parser.error("--speckle-size must be zero or positive")
    if args.speckle_diff < 0.0:
        parser.error("--speckle-diff must be zero or positive")
    return args


def set_rerun_time(rr, frame_index: int) -> None:
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("frame", frame_index)
    else:
        rr.set_time("frame", sequence=frame_index)


def init_rerun(args: argparse.Namespace):
    import rerun as rr

    rr.init("dv_stereo_pointcloud")
    if args.save_rrd is not None:
        rr.save(str(args.save_rrd))
    elif args.connect:
        if hasattr(rr, "connect_grpc"):
            rr.connect_grpc()
        else:
            rr.connect()
    else:
        rr.spawn(memory_limit=args.rerun_memory_limit, server_memory_limit=args.rerun_server_memory_limit)
    return rr


def disparity_preview(disparity: np.ndarray, min_disparity: float) -> np.ndarray:
    disparity_px = np.asarray(disparity, dtype=np.float32) / 16.0
    valid = disparity_px > min_disparity
    preview = np.zeros(disparity_px.shape, dtype=np.uint8)
    if np.any(valid):
        preview[valid] = cv.normalize(disparity_px[valid], None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1).reshape(-1)
    colored = cv.applyColorMap(preview, cv.COLORMAP_JET)
    colored[~valid] = 0
    return cv.cvtColor(colored, cv.COLOR_BGR2RGB)


def filter_disparity(disparity: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    filtered = np.asarray(disparity).copy()
    if args.median_filter_size > 1:
        filtered = cv.medianBlur(filtered, args.median_filter_size)
    if args.speckle_size > 0:
        cv.filterSpeckles(filtered, 0, args.speckle_size, int(round(args.speckle_diff * 16.0)))
    return filtered


def disparity_stats(disparity: np.ndarray, *, min_disparity: float, max_depth_m: float, stereo_geometry) -> dict:
    disparity_px = np.asarray(disparity, dtype=np.float32) / 16.0
    positive = disparity_px > 0.0
    above_min = disparity_px > min_disparity
    stats = {
        "positive": int(np.count_nonzero(positive)),
        "above_min": int(np.count_nonzero(above_min)),
        "total": int(disparity_px.size),
    }
    if np.any(positive):
        values = disparity_px[positive]
        stats["disp_min"] = float(values.min())
        stats["disp_mean"] = float(values.mean())
        stats["disp_max"] = float(values.max())
    if np.any(above_min):
        focal_baseline_mm = float(stereo_geometry.convertDisparityToDepth(1.0))
        depth_m = (focal_baseline_mm / disparity_px[above_min]) / 1000.0
        depth_m = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
        if len(depth_m) > 0:
            stats["depth_min_m"] = float(depth_m.min())
            stats["depth_mean_m"] = float(depth_m.mean())
            stats["depth_max_m"] = float(depth_m.max())
            stats["within_depth"] = int(np.count_nonzero(depth_m <= max_depth_m))
    return stats


def print_calibration_quality_hint(calibration_file: Path) -> None:
    summary_file = calibration_file.with_name("summary.json")
    if not summary_file.exists():
        return

    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    left_error = summary.get("left_reprojection_error")
    right_error = summary.get("right_reprojection_error")
    epipolar_error = summary.get("mean_epipolar_error")
    print(
        "Calibration quality: "
        f"left_reprojection_error={left_error}, "
        f"right_reprojection_error={right_error}, "
        f"mean_epipolar_error={epipolar_error}"
    )
    if (
        isinstance(left_error, (int, float))
        and isinstance(right_error, (int, float))
        and isinstance(epipolar_error, (int, float))
        and (left_error > 1.0 or right_error > 1.0 or epipolar_error > 1.0)
    ):
        print(
            "WARNING: Calibration error is high for dense stereo. "
            "If disparity looks fragmented, recalibrate with more varied board poses and better corner coverage."
        )


class DisparityProjector:
    def __init__(self, stereo_geometry: Any):
        left_geometry = stereo_geometry.getLeftCameraGeometry()
        camera_matrix = left_geometry.getCameraMatrix()
        self.fx = float(camera_matrix[0, 0])
        self.fy = float(camera_matrix[1, 1])
        self.cx = float(camera_matrix[0, 2])
        self.cy = float(camera_matrix[1, 2])
        self.focal_baseline_mm = float(stereo_geometry.convertDisparityToDepth(1.0))
        self._shape: tuple[int, int] | None = None
        self._x_factor: np.ndarray | None = None
        self._y_factor: np.ndarray | None = None

    def factors(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        if self._shape != shape:
            ys, xs = np.indices(shape, dtype=np.float32)
            self._x_factor = ((xs - self.cx) / self.fx).reshape(-1)
            self._y_factor = ((ys - self.cy) / self.fy).reshape(-1)
            self._shape = shape
        return self._x_factor, self._y_factor


def pointcloud_from_disparity(
    disparity: np.ndarray,
    left_image: np.ndarray,
    projector: DisparityProjector,
    *,
    min_disparity: float,
    max_depth_m: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    disparity_px = np.asarray(disparity, dtype=np.float32) / 16.0
    disparity_flat = disparity_px.reshape(-1)
    valid_indices = np.flatnonzero(disparity_flat > min_disparity)
    if len(valid_indices) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    depth_m = (projector.focal_baseline_mm / disparity_flat[valid_indices]) / 1000.0
    valid_depth = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= max_depth_m)
    if not np.any(valid_depth):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    valid_indices = valid_indices[valid_depth]
    depth_m = depth_m[valid_depth].astype(np.float32, copy=False)

    if max_points > 0 and len(valid_indices) > max_points:
        step = int(np.ceil(len(valid_indices) / max_points))
        selected = np.arange(0, len(valid_indices), step, dtype=np.int64)
        valid_indices = valid_indices[selected]
        depth_m = depth_m[selected]

    x_factor, y_factor = projector.factors(disparity_px.shape)
    points_m = np.empty((len(valid_indices), 3), dtype=np.float32)
    points_m[:, 0] = x_factor[valid_indices] * depth_m
    points_m[:, 1] = y_factor[valid_indices] * depth_m
    points_m[:, 2] = depth_m

    if left_image.ndim == 2:
        values = left_image.reshape(-1)[valid_indices].reshape(-1, 1)
        colors = np.repeat(values, 3, axis=1).astype(np.uint8)
    else:
        colors = cv.cvtColor(left_image, cv.COLOR_BGR2RGB).reshape(-1, 3)[valid_indices].astype(np.uint8)

    return points_m, colors


def main() -> None:
    args = parse_args()
    rr = init_rerun(args)

    calibration = load_stereo_calibration(
        args.calibration,
        args.left_designation,
        args.right_designation,
    )
    print_calibration_quality_hint(args.calibration)
    pair = open_stereo_cameras(
        args.left or calibration.left.name,
        args.right or calibration.right.name,
    )
    stereo_geometry = dv.camera.StereoGeometry(calibration.left, calibration.right)
    left_accumulator = dv.EdgeMapAccumulator(
        calibration.left.resolution,
        args.matcher_contribution,
        args.matcher_ignore_polarity,
        0.0,
        args.matcher_decay,
    )
    right_accumulator = dv.EdgeMapAccumulator(
        calibration.right.resolution,
        args.matcher_contribution,
        args.matcher_ignore_polarity,
        0.0,
        args.matcher_decay,
    )
    matcher = dv.SemiDenseStereoMatcher(stereo_geometry, left_accumulator, right_accumulator)
    projector = DisparityProjector(stereo_geometry)

    event_count = args.event_count
    if event_count is None:
        width, height = calibration.left.resolution
        event_count = int((width * height) / 3)

    slicer = dv.StereoEventStreamSlicer()
    left_buffer = dv.EventStore()
    right_buffer = dv.EventStore()
    frame_index = 0

    def callback(left_events, right_events) -> None:
        nonlocal left_buffer, right_buffer, frame_index
        left_buffer.add(left_events)
        right_buffer.add(right_events)
        if len(left_buffer) > event_count:
            left_buffer = left_buffer.sliceBack(event_count)
        if len(right_buffer) > event_count:
            right_buffer = right_buffer.sliceBack(event_count)
        if len(left_buffer) == 0 or len(right_buffer) == 0:
            return

        raw_disparity = matcher.computeDisparity(left_buffer, right_buffer)
        disparity = filter_disparity(raw_disparity, args)
        left_image = matcher.getLeftFrame().image
        right_image = matcher.getRightFrame().image
        should_log_images = args.image_interval > 0 and frame_index % args.image_interval == 0
        should_log_pointcloud = args.pointcloud_interval > 0 and frame_index % args.pointcloud_interval == 0

        set_rerun_time(rr, frame_index)
        if should_log_images:
            rr.log("stereo/left_accumulated", rr.Image(left_image))
            rr.log("stereo/right_accumulated", rr.Image(right_image))
            rr.log("stereo/disparity", rr.Image(disparity_preview(disparity, args.min_disparity)))

        point_count: int | None = None
        if should_log_pointcloud:
            points, colors = pointcloud_from_disparity(
                disparity,
                left_image,
                projector,
                min_disparity=args.min_disparity,
                max_depth_m=args.max_depth_m,
                max_points=args.max_points,
            )
            point_count = len(points)
            if point_count > 0:
                rr.log("stereo/pointcloud", rr.Points3D(points, colors=colors, radii=args.point_radius))
            else:
                rr.log("stereo/pointcloud", rr.Clear(recursive=False))

        if args.stats_interval > 0 and frame_index % args.stats_interval == 0:
            stats = disparity_stats(
                disparity,
                min_disparity=args.min_disparity,
                max_depth_m=args.max_depth_m,
                stereo_geometry=stereo_geometry,
            )
            print(
                "frame={frame} left_events={left_events} right_events={right_events} "
                "positive_disp={positive}/{total} above_min={above_min} logged_points={points} stats={stats}".format(
                    frame=frame_index,
                    left_events=len(left_buffer),
                    right_events=len(right_buffer),
                    points=point_count if point_count is not None else "skipped",
                    **stats,
                    stats=stats,
                )
            )
        frame_index += 1

    slicer.doEveryTimeInterval(slicer_interval(args), callback)

    while cameras_are_running(pair):
        slicer.accept(*next_stereo_events(pair))


if __name__ == "__main__":
    main()
