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
    frame_image,
    load_stereo_calibration,
    next_stereo_events,
    open_stereo_cameras,
    slicer_interval,
)


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("RGB color must be formatted as R,G,B")
    try:
        color = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("RGB color values must be integers") from exc
    if any(channel < 0 or channel > 255 for channel in color):
        raise argparse.ArgumentTypeError("RGB color values must be in the range 0..255")
    return color


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate stereo depth from events and visualize the point cloud in Rerun.")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--opencv-calibration", type=Path, help="OpenCV .npz calibration file. Defaults to opencv_calibration.npz next to --calibration.")
    parser.add_argument("--backend", choices=("opencv", "dv"), default="opencv", help="Stereo matcher backend. OpenCV SGBM is adjustable and better for near objects.")
    parser.add_argument("--left-designation", default="C0")
    parser.add_argument("--right-designation", default="C1")
    add_camera_selection_args(parser)
    add_slicer_args(parser, default_interval_ms=100)
    parser.add_argument("--event-count", type=int, help="Overlapping event buffer size per camera.")
    parser.add_argument("--max-points", type=int, default=40_000)
    parser.add_argument("--min-disparity", type=float, default=1.0)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--point-radius", type=float, default=0.01)
    parser.add_argument("--point-color-mode", choices=("depth", "image", "fixed"), default="depth")
    parser.add_argument("--point-color", type=parse_rgb, default=parse_rgb("0,220,255"), help="RGB color used when --point-color-mode fixed.")
    parser.add_argument("--median-filter-size", type=int, default=3, help="Odd kernel size for disparity median filtering. Use 0 to disable.")
    parser.add_argument("--speckle-size", type=int, default=100, help="Remove disparity blobs up to this pixel size. Use 0 to disable.")
    parser.add_argument("--speckle-diff", type=float, default=2.0, help="Maximum disparity difference in px for speckle filtering.")
    parser.add_argument("--matcher-contribution", type=float, default=0.25, help="Event contribution for the matcher EdgeMapAccumulator.")
    parser.add_argument("--matcher-decay", type=float, default=1.0, help="Decay for the matcher EdgeMapAccumulator.")
    parser.add_argument("--use-polarity", action="store_false", dest="matcher_ignore_polarity", help="Keep event polarity in matcher accumulation.")
    parser.add_argument("--ignore-polarity", action="store_true", dest="matcher_ignore_polarity", help="Ignore event polarity in matcher accumulation.")
    parser.add_argument("--sgbm-min-disparity", type=int, default=0)
    parser.add_argument("--sgbm-num-disparities", type=int, default=192, help="OpenCV SGBM disparity search range. Must be divisible by 16.")
    parser.add_argument("--sgbm-block-size", type=int, default=5, help="OpenCV SGBM block size. Must be odd.")
    parser.add_argument("--sgbm-uniqueness-ratio", type=int, default=5)
    parser.add_argument("--sgbm-disp12-max-diff", type=int, default=1)
    parser.add_argument("--sgbm-pre-filter-cap", type=int, default=31)
    parser.add_argument("--sgbm-p1", type=int, help="OpenCV SGBM P1. Defaults to 8 * block_size^2.")
    parser.add_argument("--sgbm-p2", type=int, help="OpenCV SGBM P2. Defaults to 32 * block_size^2.")
    parser.add_argument(
        "--sgbm-mode",
        choices=("sgbm", "hh", "sgbm-3way", "hh4"),
        default="sgbm-3way",
        help="OpenCV SGBM mode.",
    )
    parser.add_argument("--pointcloud-interval", type=int, default=2, help="Log point clouds to Rerun every N frames.")
    parser.add_argument("--image-interval", type=int, default=2, help="Log stereo/disparity images to Rerun every N frames.")
    parser.add_argument("--clear-empty-pointcloud", action="store_true", help="Clear the Rerun point cloud entity when no valid points are produced.")
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
    if args.sgbm_num_disparities <= 0 or args.sgbm_num_disparities % 16 != 0:
        parser.error("--sgbm-num-disparities must be a positive multiple of 16")
    if args.sgbm_block_size <= 0 or args.sgbm_block_size % 2 == 0:
        parser.error("--sgbm-block-size must be a positive odd number")
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


def disparity_stats(disparity: np.ndarray, *, min_disparity: float, max_depth_m: float, projector) -> dict:
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
        depth_m = projector.focal_baseline_m / disparity_px[above_min]
        depth_m = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
        if len(depth_m) > 0:
            stats["depth_min_m"] = float(depth_m.min())
            stats["depth_mean_m"] = float(depth_m.mean())
            stats["depth_max_m"] = float(depth_m.max())
            stats["within_depth"] = int(np.count_nonzero(depth_m <= max_depth_m))
        else:
            stats["within_depth"] = 0
    else:
        stats["within_depth"] = 0
    return stats


def pointcloud_filter_message(stats: dict, max_depth_m: float) -> str | None:
    if stats["above_min"] == 0:
        return "no disparity values passed --min-disparity"
    if stats.get("within_depth", 0) == 0:
        return f"all candidate points are farther than --max-depth-m {max_depth_m}"
    return None


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


def opencv_calibration_path(args: argparse.Namespace) -> Path:
    return args.opencv_calibration or args.calibration.with_name("opencv_calibration.npz")


def load_opencv_calibration(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"OpenCV calibration file was not found: {path}")

    required = (
        "image_size",
        "left_matrix",
        "left_distortion",
        "right_matrix",
        "right_distortion",
        "rect_left",
        "rect_right",
        "proj_left",
        "proj_right",
    )
    with np.load(path) as data:
        missing = [name for name in required if name not in data]
        if missing:
            raise RuntimeError(f"OpenCV calibration file is missing: {', '.join(missing)}")
        calibration = {name: data[name].copy() for name in required}

    image_size_values = calibration["image_size"].reshape(-1)
    if len(image_size_values) != 2:
        raise RuntimeError("OpenCV calibration image_size must contain width and height")
    calibration["image_size"] = np.asarray(image_size_values, dtype=np.int32)
    return calibration


def create_rectification_maps(calibration: dict[str, np.ndarray]):
    width, height = (int(value) for value in calibration["image_size"])
    image_size = (width, height)
    left_map = cv.initUndistortRectifyMap(
        calibration["left_matrix"],
        calibration["left_distortion"],
        calibration["rect_left"],
        calibration["proj_left"],
        image_size,
        cv.CV_32FC1,
    )
    right_map = cv.initUndistortRectifyMap(
        calibration["right_matrix"],
        calibration["right_distortion"],
        calibration["rect_right"],
        calibration["proj_right"],
        image_size,
        cv.CV_32FC1,
    )
    return left_map, right_map


def create_sgbm(args: argparse.Namespace):
    block_size = args.sgbm_block_size
    p1 = args.sgbm_p1 if args.sgbm_p1 is not None else 8 * block_size * block_size
    p2 = args.sgbm_p2 if args.sgbm_p2 is not None else 32 * block_size * block_size
    mode_map = {
        "sgbm": cv.STEREO_SGBM_MODE_SGBM,
        "hh": cv.STEREO_SGBM_MODE_HH,
        "sgbm-3way": cv.STEREO_SGBM_MODE_SGBM_3WAY,
        "hh4": cv.STEREO_SGBM_MODE_HH4,
    }
    return cv.StereoSGBM_create(
        minDisparity=args.sgbm_min_disparity,
        numDisparities=args.sgbm_num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=args.sgbm_disp12_max_diff,
        preFilterCap=args.sgbm_pre_filter_cap,
        uniquenessRatio=args.sgbm_uniqueness_ratio,
        speckleWindowSize=0,
        speckleRange=0,
        mode=mode_map[args.sgbm_mode],
    )


def create_edge_accumulator(resolution: tuple[int, int], args: argparse.Namespace):
    return dv.EdgeMapAccumulator(
        resolution,
        args.matcher_contribution,
        args.matcher_ignore_polarity,
        0.0,
        args.matcher_decay,
    )


def compute_opencv_disparity(
    left_events,
    right_events,
    left_accumulator,
    right_accumulator,
    left_map,
    right_map,
    sgbm,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_accumulator.reset()
    right_accumulator.reset()
    left_accumulator.accept(left_events)
    right_accumulator.accept(right_events)
    left_image = frame_image(left_accumulator.generateFrame()).copy()
    right_image = frame_image(right_accumulator.generateFrame()).copy()
    left_rectified = cv.remap(left_image, left_map[0], left_map[1], cv.INTER_LINEAR)
    right_rectified = cv.remap(right_image, right_map[0], right_map[1], cv.INTER_LINEAR)
    disparity = sgbm.compute(left_rectified, right_rectified)
    return disparity, left_rectified, right_rectified


def print_depth_range(args: argparse.Namespace, projector: "DisparityProjector") -> None:
    if args.backend == "opencv":
        max_disparity = args.sgbm_min_disparity + args.sgbm_num_disparities
    else:
        max_disparity = 48
    min_depth = projector.focal_baseline_m / max_disparity
    print(
        f"Stereo backend={args.backend} fB={projector.focal_baseline_m:.6f} m*px "
        f"depth@{max_disparity}px={min_depth:.3f} m"
    )


def depth_colors(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    normalized = np.clip(depth_m / max_depth_m, 0.0, 1.0)
    values = (normalized * 255.0).astype(np.uint8).reshape(-1, 1)
    color_map = cv.COLORMAP_TURBO if hasattr(cv, "COLORMAP_TURBO") else cv.COLORMAP_JET
    return cv.cvtColor(cv.applyColorMap(values, color_map), cv.COLOR_BGR2RGB).reshape(-1, 3)


class DisparityProjector:
    def __init__(self, camera_matrix: np.ndarray, focal_baseline_m: float):
        self.fx = float(camera_matrix[0, 0])
        self.fy = float(camera_matrix[1, 1])
        self.cx = float(camera_matrix[0, 2])
        self.cy = float(camera_matrix[1, 2])
        self.focal_baseline_m = float(focal_baseline_m)
        self._shape: tuple[int, int] | None = None
        self._x_factor: np.ndarray | None = None
        self._y_factor: np.ndarray | None = None

    @classmethod
    def from_dv(cls, stereo_geometry: Any):
        left_geometry = stereo_geometry.getLeftCameraGeometry()
        camera_matrix = left_geometry.getCameraMatrix()
        focal_baseline_m = float(stereo_geometry.convertDisparityToDepth(1.0)) / 1000.0
        return cls(camera_matrix, focal_baseline_m)

    @classmethod
    def from_opencv(cls, calibration: dict[str, np.ndarray]):
        proj_left = calibration["proj_left"]
        proj_right = calibration["proj_right"]
        camera_matrix = np.asarray(
            [
                [proj_left[0, 0], 0.0, proj_left[0, 2]],
                [0.0, proj_left[1, 1], proj_left[1, 2]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        focal_baseline_m = abs(float(proj_right[0, 3]))
        return cls(camera_matrix, focal_baseline_m)

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
    color_mode: str,
    fixed_color: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    disparity_px = np.asarray(disparity, dtype=np.float32) / 16.0
    disparity_flat = disparity_px.reshape(-1)
    valid_indices = np.flatnonzero(disparity_flat > min_disparity)
    if len(valid_indices) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    depth_m = projector.focal_baseline_m / disparity_flat[valid_indices]
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

    if color_mode == "depth":
        colors = depth_colors(depth_m, max_depth_m)
    elif color_mode == "fixed":
        colors = np.tile(np.asarray(fixed_color, dtype=np.uint8), (len(valid_indices), 1))
    elif left_image.ndim == 2:
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
    if args.backend == "opencv":
        opencv_file = opencv_calibration_path(args)
        opencv_calibration = load_opencv_calibration(opencv_file)
        left_map, right_map = create_rectification_maps(opencv_calibration)
        left_accumulator = create_edge_accumulator(calibration.left.resolution, args)
        right_accumulator = create_edge_accumulator(calibration.right.resolution, args)
        sgbm = create_sgbm(args)
        matcher = None
        projector = DisparityProjector.from_opencv(opencv_calibration)
        print(f"Using OpenCV calibration: {opencv_file}")
    else:
        stereo_geometry = dv.camera.StereoGeometry(calibration.left, calibration.right)
        left_accumulator = create_edge_accumulator(calibration.left.resolution, args)
        right_accumulator = create_edge_accumulator(calibration.right.resolution, args)
        matcher = dv.SemiDenseStereoMatcher(stereo_geometry, left_accumulator, right_accumulator)
        left_map = right_map = sgbm = None
        projector = DisparityProjector.from_dv(stereo_geometry)
    print_depth_range(args, projector)

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

        if args.backend == "opencv":
            raw_disparity, left_image, right_image = compute_opencv_disparity(
                left_buffer,
                right_buffer,
                left_accumulator,
                right_accumulator,
                left_map,
                right_map,
                sgbm,
            )
        else:
            raw_disparity = matcher.computeDisparity(left_buffer, right_buffer)
            left_image = matcher.getLeftFrame().image
            right_image = matcher.getRightFrame().image
        disparity = filter_disparity(raw_disparity, args)
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
                color_mode=args.point_color_mode,
                fixed_color=args.point_color,
            )
            point_count = len(points)
            if point_count > 0:
                rr.log("stereo/pointcloud", rr.Points3D(points, colors=colors, radii=args.point_radius))
            elif args.clear_empty_pointcloud:
                rr.log("stereo/pointcloud", rr.Clear(recursive=False))

        if args.stats_interval > 0 and frame_index % args.stats_interval == 0:
            stats = disparity_stats(
                disparity,
                min_disparity=args.min_disparity,
                max_depth_m=args.max_depth_m,
                projector=projector,
            )
            message = pointcloud_filter_message(stats, args.max_depth_m)
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
            if message is not None:
                print(f"Point cloud is empty or sparse: {message}")
        frame_index += 1

    slicer.doEveryTimeInterval(slicer_interval(args), callback)

    while cameras_are_running(pair):
        slicer.accept(*next_stereo_events(pair))


if __name__ == "__main__":
    main()
