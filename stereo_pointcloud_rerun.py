from __future__ import annotations

import argparse
from pathlib import Path

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
    add_slicer_args(parser)
    parser.add_argument("--event-count", type=int, help="Overlapping event buffer size per camera.")
    parser.add_argument("--max-points", type=int, default=120_000)
    parser.add_argument("--min-disparity", type=float, default=1.0)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--point-radius", type=float, default=0.003)
    parser.add_argument("--stats-interval", type=int, default=30, help="Print disparity/point-cloud stats every N frames.")
    parser.add_argument("--connect", action="store_true", help="Connect to an already running Rerun viewer.")
    parser.add_argument("--save-rrd", type=Path, help="Write a Rerun recording instead of spawning a viewer.")
    return parser.parse_args()


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
        rr.spawn()
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


def pointcloud_from_disparity(
    disparity: np.ndarray,
    left_image: np.ndarray,
    stereo_geometry,
    *,
    min_disparity: float,
    max_depth_m: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    disparity_px = np.asarray(disparity, dtype=np.float32) / 16.0
    valid = disparity_px > min_disparity
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    left_geometry = stereo_geometry.getLeftCameraGeometry()
    camera_matrix = left_geometry.getCameraMatrix()
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    focal_baseline_mm = float(stereo_geometry.convertDisparityToDepth(1.0))

    z_mm = np.zeros_like(disparity_px, dtype=np.float32)
    z_mm[valid] = focal_baseline_mm / disparity_px[valid]
    valid &= np.isfinite(z_mm) & (z_mm > 0.0) & (z_mm <= max_depth_m * 1000.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    ys, xs = np.indices(disparity_px.shape, dtype=np.float32)
    x_mm = (xs - cx) * z_mm / fx
    y_mm = (ys - cy) * z_mm / fy
    points_m = np.stack((x_mm, y_mm, z_mm), axis=-1)[valid] / 1000.0

    if left_image.ndim == 2:
        values = left_image[valid].reshape(-1, 1)
        colors = np.repeat(values, 3, axis=1).astype(np.uint8)
    else:
        colors = cv.cvtColor(left_image, cv.COLOR_BGR2RGB)[valid].astype(np.uint8)

    if max_points > 0 and len(points_m) > max_points:
        step = int(np.ceil(len(points_m) / max_points))
        points_m = points_m[::step]
        colors = colors[::step]

    return points_m.astype(np.float32), colors


def main() -> None:
    args = parse_args()
    rr = init_rerun(args)

    calibration = load_stereo_calibration(
        args.calibration,
        args.left_designation,
        args.right_designation,
    )
    pair = open_stereo_cameras(
        args.left or calibration.left.name,
        args.right or calibration.right.name,
    )
    stereo_geometry = dv.camera.StereoGeometry(calibration.left, calibration.right)
    matcher = dv.SemiDenseStereoMatcher(stereo_geometry)

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

        disparity = matcher.computeDisparity(left_buffer, right_buffer)
        left_image = matcher.getLeftFrame().image
        right_image = matcher.getRightFrame().image
        points, colors = pointcloud_from_disparity(
            disparity,
            left_image,
            stereo_geometry,
            min_disparity=args.min_disparity,
            max_depth_m=args.max_depth_m,
            max_points=args.max_points,
        )

        set_rerun_time(rr, frame_index)
        rr.log("stereo/left_accumulated", rr.Image(left_image))
        rr.log("stereo/right_accumulated", rr.Image(right_image))
        rr.log("stereo/disparity", rr.Image(disparity_preview(disparity, args.min_disparity)))
        if len(points) > 0:
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
                "positive_disp={positive}/{total} above_min={above_min} points={points} stats={stats}".format(
                    frame=frame_index,
                    left_events=len(left_buffer),
                    right_events=len(right_buffer),
                    points=len(points),
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
