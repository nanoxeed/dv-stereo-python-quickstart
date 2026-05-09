from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import dv_processing as dv
import numpy as np

from stereo_common import (
    add_accumulator_args,
    add_camera_selection_args,
    add_slicer_args,
    cameras_are_running,
    create_stereo_accumulators,
    ensure_bgr,
    ensure_gray,
    frame_image,
    next_stereo_events,
    open_stereo_cameras,
    side_by_side,
    slicer_interval,
    write_dv_stereo_calibration,
)


@dataclass
class CalibrationSample:
    object_points: np.ndarray
    left_points: np.ndarray
    right_points: np.ndarray
    left_image: np.ndarray
    right_image: np.ndarray


def parse_pattern_size(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(",", "x")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Pattern size must be formatted as COLSxROWS, for example 8x5")
    cols, rows = int(parts[0]), int(parts[1])
    if cols <= 0 or rows <= 0:
        raise argparse.ArgumentTypeError("Pattern dimensions must be positive")
    return cols, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate synchronized stereo event cameras from accumulated images.")
    add_camera_selection_args(parser)
    add_slicer_args(parser)
    add_accumulator_args(parser, default_kind="generic")
    parser.set_defaults(contribution=0.05, decay=1.0e6, generic_decay="exponential", max_potential=0.3)
    parser.add_argument("--pattern", choices=("chessboard", "circles", "asymmetric-circles"), default="chessboard")
    parser.add_argument(
        "--pattern-size",
        type=parse_pattern_size,
        default=parse_pattern_size("8x5"),
        help="Detected points as COLSxROWS. For a 9x6 chessboard of squares, use 8x5 inner corners.",
    )
    parser.add_argument("--square-size", type=float, default=30.0, help="Pattern spacing before unit conversion.")
    parser.add_argument(
        "--square-size-scale-to-meters",
        type=float,
        default=0.001,
        help="Scale applied to --square-size before calibration. The default treats square size as millimeters.",
    )
    parser.add_argument("--min-detections", type=int, default=20)
    parser.add_argument("--consecutive-detections", type=int, default=3)
    parser.add_argument("--sample-cooldown-sec", type=float, default=0.35)
    parser.add_argument("--output-dir", type=Path, default=Path("calibration"))
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--no-review", action="store_true")
    parser.add_argument("--refine-intrinsics", action="store_true")
    parser.add_argument("--max-reprojection-error", type=float, default=1.0)
    parser.add_argument("--max-epipolar-error", type=float, default=1.0)
    parser.add_argument("--enforce-error-limits", action="store_true")
    parser.add_argument("--left-name", help="Override left camera name written into the calibration file.")
    parser.add_argument("--right-name", help="Override right camera name written into the calibration file.")
    parser.add_argument("--window-name", default="Stereo calibration")
    return parser.parse_args()


def make_object_points(pattern: str, pattern_size: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    points = np.zeros((rows * cols, 3), np.float32)

    if pattern == "asymmetric-circles":
        coords = []
        for row in range(rows):
            for col in range(cols):
                coords.append(((2 * col + row % 2) * square_size, row * square_size, 0.0))
        points[:] = np.asarray(coords, dtype=np.float32)
        return points

    grid_x, grid_y = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
    points[:, 0] = grid_x.reshape(-1) * square_size
    points[:, 1] = grid_y.reshape(-1) * square_size
    return points


def candidate_images(image: np.ndarray) -> list[np.ndarray]:
    gray = ensure_gray(image)
    candidates = [gray]
    if gray.dtype == np.uint8:
        equalized = cv.equalizeHist(gray)
        candidates.extend([equalized, cv.bitwise_not(gray), cv.bitwise_not(equalized)])
    return candidates


def detect_pattern(image: np.ndarray, pattern: str, pattern_size: tuple[int, int]):
    if pattern == "chessboard":
        sb_flags = cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY | cv.CALIB_CB_NORMALIZE_IMAGE
        for candidate in candidate_images(image):
            ok, corners = cv.findChessboardCornersSB(candidate, pattern_size, sb_flags)
            if ok:
                return True, corners.astype(np.float32)

        fallback_flags = cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        for candidate in candidate_images(image):
            ok, corners = cv.findChessboardCorners(candidate, pattern_size, fallback_flags)
            if ok:
                corners = cv.cornerSubPix(candidate, corners, (5, 5), (-1, -1), criteria)
                return True, corners.astype(np.float32)
        return False, None

    flags = cv.CALIB_CB_SYMMETRIC_GRID if pattern == "circles" else cv.CALIB_CB_ASYMMETRIC_GRID
    flags |= cv.CALIB_CB_CLUSTERING
    for candidate in candidate_images(image):
        ok, centers = cv.findCirclesGrid(candidate, pattern_size, flags=flags)
        if ok:
            return True, centers.astype(np.float32)
    return False, None


def draw_detection(image: np.ndarray, pattern_size: tuple[int, int], points, found: bool) -> np.ndarray:
    preview = ensure_bgr(image).copy()
    if points is not None:
        cv.drawChessboardCorners(preview, pattern_size, points, found)
    return preview


def add_coverage(coverage: np.ndarray, points: np.ndarray) -> None:
    hull = cv.convexHull(points.reshape(-1, 2).astype(np.float32)).astype(np.int32)
    cv.fillConvexPoly(coverage, hull, 255)


def overlay_coverage(image: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    out = ensure_bgr(image).copy()
    green = np.zeros_like(out)
    green[:, :, 1] = 180
    mask = coverage > 0
    out[mask] = cv.addWeighted(out, 0.55, green, 0.45, 0)[mask]
    return out


def review_samples(samples: list[CalibrationSample], pattern_size: tuple[int, int]) -> list[CalibrationSample]:
    kept: list[CalibrationSample] = []
    for index, sample in enumerate(samples, start=1):
        left = draw_detection(sample.left_image, pattern_size, sample.left_points, True)
        right = draw_detection(sample.right_image, pattern_size, sample.right_points, True)
        preview = side_by_side(left, right, f"Sample {index} left", f"Sample {index} right")
        cv.putText(
            preview,
            "space/k: keep    d: discard    esc: stop review",
            (12, preview.shape[0] - 14),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )

        while True:
            cv.imshow("Review calibration samples", preview)
            key = cv.waitKey(0) & 0xFF
            if key in (ord(" "), ord("k"), 13):
                kept.append(sample)
                break
            if key == ord("d"):
                break
            if key == 27:
                cv.destroyWindow("Review calibration samples")
                return kept
    cv.destroyWindow("Review calibration samples")
    return kept


def reprojection_error(object_points, image_points, rvecs, tvecs, matrix, distortion) -> float:
    total_error = 0.0
    total_points = 0
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv.projectPoints(obj, rvec, tvec, matrix, distortion)
        error = cv.norm(img, projected, cv.NORM_L2)
        total_error += error * error
        total_points += len(projected)
    return math.sqrt(total_error / total_points)


def epipolar_error(left_points, right_points, fundamental: np.ndarray) -> float:
    total = 0.0
    count = 0
    for left, right in zip(left_points, right_points):
        left_xy = left.reshape(-1, 2)
        right_xy = right.reshape(-1, 2)
        right_lines = cv.computeCorrespondEpilines(left_xy.reshape(-1, 1, 2), 1, fundamental).reshape(-1, 3)
        left_lines = cv.computeCorrespondEpilines(right_xy.reshape(-1, 1, 2), 2, fundamental).reshape(-1, 3)

        right_den = np.linalg.norm(right_lines[:, :2], axis=1)
        left_den = np.linalg.norm(left_lines[:, :2], axis=1)
        right_distance = np.abs(np.sum(right_lines[:, :2] * right_xy, axis=1) + right_lines[:, 2]) / right_den
        left_distance = np.abs(np.sum(left_lines[:, :2] * left_xy, axis=1) + left_lines[:, 2]) / left_den
        total += float(np.sum(right_distance + left_distance))
        count += len(left_xy) * 2
    return total / count


def calibrate(samples: list[CalibrationSample], image_size: tuple[int, int], args: argparse.Namespace, pair) -> dict:
    object_points = [sample.object_points for sample in samples]
    left_points = [sample.left_points for sample in samples]
    right_points = [sample.right_points for sample in samples]

    mono_flags = 0
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    left_rms, left_matrix, left_distortion, left_rvecs, left_tvecs = cv.calibrateCamera(
        object_points, left_points, image_size, None, None, flags=mono_flags, criteria=criteria
    )
    right_rms, right_matrix, right_distortion, right_rvecs, right_tvecs = cv.calibrateCamera(
        object_points, right_points, image_size, None, None, flags=mono_flags, criteria=criteria
    )

    stereo_flags = 0 if args.refine_intrinsics else cv.CALIB_FIX_INTRINSIC
    (
        stereo_rms,
        left_matrix,
        left_distortion,
        right_matrix,
        right_distortion,
        rotation,
        translation,
        essential,
        fundamental,
    ) = cv.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        left_matrix,
        left_distortion,
        right_matrix,
        right_distortion,
        image_size,
        criteria=criteria,
        flags=stereo_flags,
    )

    left_error = reprojection_error(object_points, left_points, left_rvecs, left_tvecs, left_matrix, left_distortion)
    right_error = reprojection_error(object_points, right_points, right_rvecs, right_tvecs, right_matrix, right_distortion)
    stereo_epipolar_error = epipolar_error(left_points, right_points, fundamental)

    rect_left, rect_right, proj_left, proj_right, q_matrix, _, _ = cv.stereoRectify(
        left_matrix,
        left_distortion,
        right_matrix,
        right_distortion,
        image_size,
        rotation,
        translation,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dv_file = args.output_dir / "stereo_calibration.json"
    left_name = args.left_name or pair.left_name
    right_name = args.right_name or pair.right_name
    write_dv_stereo_calibration(
        dv_file,
        left_name=left_name,
        right_name=right_name,
        image_size=image_size,
        left_matrix=left_matrix,
        left_distortion=left_distortion,
        right_matrix=right_matrix,
        right_distortion=right_distortion,
        rotation_left_to_right=rotation,
        translation_left_to_right=translation,
        essential=essential,
        fundamental=fundamental,
    )

    np.savez(
        args.output_dir / "opencv_calibration.npz",
        image_size=np.asarray(image_size, dtype=np.int32),
        left_matrix=left_matrix,
        left_distortion=left_distortion,
        right_matrix=right_matrix,
        right_distortion=right_distortion,
        rotation=rotation,
        translation=translation,
        essential=essential,
        fundamental=fundamental,
        rect_left=rect_left,
        rect_right=rect_right,
        proj_left=proj_left,
        proj_right=proj_right,
        q_matrix=q_matrix,
    )

    summary = {
        "samples": len(samples),
        "image_size": list(image_size),
        "left_camera": left_name,
        "right_camera": right_name,
        "square_size": float(args.square_size),
        "square_size_scale_to_meters": float(args.square_size_scale_to_meters),
        "left_rms": float(left_rms),
        "right_rms": float(right_rms),
        "stereo_rms": float(stereo_rms),
        "left_reprojection_error": float(left_error),
        "right_reprojection_error": float(right_error),
        "mean_epipolar_error": float(stereo_epipolar_error),
        "dv_calibration": str(dv_file),
        "opencv_calibration": str(args.output_dir / "opencv_calibration.npz"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.save_images:
        image_dir = args.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples):
            cv.imwrite(str(image_dir / f"{index:03d}_left.png"), sample.left_image)
            cv.imwrite(str(image_dir / f"{index:03d}_right.png"), sample.right_image)

    return summary


def main() -> None:
    args = parse_args()
    if args.min_detections < 3:
        raise RuntimeError("--min-detections should be at least 3")

    pair = open_stereo_cameras(args.left, args.right)
    left_accumulator, right_accumulator = create_stereo_accumulators(pair, args)
    object_points_template = make_object_points(
        args.pattern,
        args.pattern_size,
        args.square_size * args.square_size_scale_to_meters,
    )

    samples: list[CalibrationSample] = []
    coverage_left = None
    coverage_right = None
    consecutive = 0
    last_sample_time = 0.0
    aborted = False

    cv.namedWindow(args.window_name, cv.WINDOW_NORMAL)
    slicer = dv.StereoEventStreamSlicer()

    def callback(left_events, right_events) -> None:
        nonlocal coverage_left, coverage_right, consecutive, last_sample_time
        left_accumulator.accept(left_events)
        right_accumulator.accept(right_events)
        left_image = frame_image(left_accumulator.generateFrame()).copy()
        right_image = frame_image(right_accumulator.generateFrame()).copy()

        if left_image.shape[:2] != right_image.shape[:2]:
            raise RuntimeError("This script currently expects both cameras to have the same accumulated image size")

        if coverage_left is None:
            coverage_left = np.zeros(left_image.shape[:2], dtype=np.uint8)
            coverage_right = np.zeros(right_image.shape[:2], dtype=np.uint8)

        left_ok, left_points = detect_pattern(left_image, args.pattern, args.pattern_size)
        right_ok, right_points = detect_pattern(right_image, args.pattern, args.pattern_size)

        if left_ok and right_ok:
            consecutive += 1
        else:
            consecutive = 0

        now = time.monotonic()
        if (
            left_ok
            and right_ok
            and consecutive >= args.consecutive_detections
            and len(samples) < args.min_detections
            and now - last_sample_time >= args.sample_cooldown_sec
        ):
            samples.append(
                CalibrationSample(
                    object_points=object_points_template.copy(),
                    left_points=left_points.copy(),
                    right_points=right_points.copy(),
                    left_image=left_image,
                    right_image=right_image,
                )
            )
            add_coverage(coverage_left, left_points)
            add_coverage(coverage_right, right_points)
            last_sample_time = now
            consecutive = 0
            print(f"Collected calibration sample {len(samples)}/{args.min_detections}")

        left_preview = draw_detection(overlay_coverage(left_image, coverage_left), args.pattern_size, left_points, left_ok)
        right_preview = draw_detection(overlay_coverage(right_image, coverage_right), args.pattern_size, right_points, right_ok)
        preview = side_by_side(
            left_preview,
            right_preview,
            f"Left detected={left_ok}",
            f"Right detected={right_ok} samples={len(samples)}/{args.min_detections}",
        )
        cv.imshow(args.window_name, preview)

    slicer.doEveryTimeInterval(slicer_interval(args), callback)

    while len(samples) < args.min_detections and cameras_are_running(pair):
        slicer.accept(*next_stereo_events(pair))
        if cv.waitKey(1) == 27:
            aborted = True
            break

    cv.destroyWindow(args.window_name)
    if aborted or len(samples) < args.min_detections:
        print(f"Calibration aborted with {len(samples)} collected samples")
        return

    if not args.no_review:
        samples = review_samples(samples, args.pattern_size)
        if len(samples) < args.min_detections:
            raise RuntimeError(f"Only {len(samples)} samples kept after review; rerun calibration or use --no-review")

    image_size = (samples[0].left_image.shape[1], samples[0].left_image.shape[0])
    summary = calibrate(samples, image_size, args, pair)

    passed = (
        summary["left_reprojection_error"] <= args.max_reprojection_error
        and summary["right_reprojection_error"] <= args.max_reprojection_error
        and summary["mean_epipolar_error"] <= args.max_epipolar_error
    )
    print(json.dumps(summary, indent=2))
    if not passed:
        message = "Calibration completed, but one or more error limits were exceeded"
        if args.enforce_error_limits:
            raise RuntimeError(message)
        print(message)


if __name__ == "__main__":
    main()
