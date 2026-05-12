from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import cv2 as cv
import dv_processing as dv
import numpy as np


@dataclass
class StereoCameraPair:
    left: Any
    right: Any
    left_name: str
    right_name: str


@dataclass
class StereoCalibrationPair:
    calibration_set: Any
    left: Any
    right: Any


def add_camera_selection_args(parser) -> None:
    parser.add_argument("--left", help="Left camera discovery index or camera name.")
    parser.add_argument("--right", help="Right camera discovery index or camera name.")


def add_accumulator_args(parser, *, default_kind: str = "edge", default_ignore_polarity: bool = True) -> None:
    parser.add_argument("--accumulator", choices=("edge", "generic"), default=default_kind)
    parser.add_argument("--contribution", type=float, default=0.25)
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--neutral", type=float, default=0.0)
    if default_ignore_polarity:
        parser.add_argument("--ignore-polarity", action="store_true", default=True)
        parser.add_argument("--use-polarity", action="store_false", dest="ignore_polarity")
    else:
        parser.add_argument("--ignore-polarity", action="store_true", default=False)
    parser.add_argument("--generic-decay", choices=("none", "linear", "exponential", "step"), default="exponential")
    parser.add_argument("--min-potential", type=float, default=0.0)
    parser.add_argument("--max-potential", type=float, default=1.0)
    parser.add_argument("--synchronous-decay", action="store_true")


def add_slicer_args(parser, *, default_interval_ms: int = 33) -> None:
    parser.add_argument("--interval-ms", type=int, default=default_interval_ms)


def open_stereo_cameras(left_selector: str | None = None, right_selector: str | None = None) -> StereoCameraPair:
    cameras = dv.io.camera.discover()

    def open_selected(selector: str | None, default_index: int):
        if selector is None:
            if len(cameras) <= default_index:
                raise RuntimeError("Unable to discover two cameras")
            return dv.io.camera.openSync(cameras[default_index])
        if selector.isdecimal():
            index = int(selector)
            if len(cameras) <= index:
                raise RuntimeError(f"Camera index {index} was requested, but only {len(cameras)} cameras were found")
            return dv.io.camera.openSync(cameras[index])
        try:
            return dv.io.camera.openSync(selector)
        except RuntimeError:
            if "_" in selector:
                return dv.io.camera.openSync(selector.rsplit("_", 1)[-1])
            raise

    left = open_selected(left_selector, 0)
    right = open_selected(right_selector, 1)
    dv.io.camera.synchronizeAnyTwo(left, right)
    ensure_event_streams(left, right)
    return StereoCameraPair(left=left, right=right, left_name=get_camera_name(left), right_name=get_camera_name(right))


def open_stereo_cameras_from_calibration(
    calibration_file: str | Path,
    left_designation: str = "C0",
    right_designation: str = "C1",
) -> tuple[StereoCameraPair, StereoCalibrationPair]:
    calibration = load_stereo_calibration(calibration_file, left_designation, right_designation)
    pair = open_stereo_cameras(calibration.left.name, calibration.right.name)
    return pair, calibration


def ensure_event_streams(left_camera, right_camera) -> None:
    if not left_camera.isEventStreamAvailable() or not right_camera.isEventStreamAvailable():
        raise RuntimeError("Both cameras must provide event streams")


def get_camera_name(camera) -> str:
    try:
        return camera.getCameraName()
    except Exception:
        return "unknown"


def next_event_store(camera):
    return camera.getNextEventBatch() or dv.EventStore()


def next_stereo_events(pair: StereoCameraPair):
    return next_event_store(pair.left), next_event_store(pair.right)


def cameras_are_running(pair: StereoCameraPair) -> bool:
    return pair.left.isRunning() and pair.right.isRunning()


def create_accumulator(
    resolution,
    *,
    kind: str = "edge",
    contribution: float = 0.25,
    decay: float = 1.0,
    neutral: float = 0.0,
    ignore_polarity: bool = True,
    generic_decay: str = "exponential",
    min_potential: float = 0.0,
    max_potential: float = 1.0,
    synchronous_decay: bool = False,
):
    if kind == "edge":
        accumulator = dv.EdgeMapAccumulator(resolution)
        accumulator.setEventContribution(contribution)
        accumulator.setDecay(decay)
        accumulator.setNeutralPotential(neutral)
        accumulator.setIgnorePolarity(ignore_polarity)
        return accumulator

    accumulator = dv.Accumulator(resolution)
    decay_map = {
        "none": dv.Accumulator.Decay.NONE,
        "linear": dv.Accumulator.Decay.LINEAR,
        "exponential": dv.Accumulator.Decay.EXPONENTIAL,
        "step": dv.Accumulator.Decay.STEP,
    }
    accumulator.setDecayFunction(decay_map[generic_decay])
    accumulator.setDecayParam(decay)
    accumulator.setEventContribution(contribution)
    accumulator.setNeutralPotential(neutral)
    accumulator.setMinPotential(min_potential)
    accumulator.setMaxPotential(max_potential)
    accumulator.setIgnorePolarity(ignore_polarity)
    accumulator.setSynchronousDecay(synchronous_decay)
    return accumulator


def create_stereo_accumulators(pair: StereoCameraPair, args):
    left = create_accumulator(
        pair.left.getEventResolution(),
        kind=args.accumulator,
        contribution=args.contribution,
        decay=args.decay,
        neutral=args.neutral,
        ignore_polarity=args.ignore_polarity,
        generic_decay=args.generic_decay,
        min_potential=args.min_potential,
        max_potential=args.max_potential,
        synchronous_decay=args.synchronous_decay,
    )
    right = create_accumulator(
        pair.right.getEventResolution(),
        kind=args.accumulator,
        contribution=args.contribution,
        decay=args.decay,
        neutral=args.neutral,
        ignore_polarity=args.ignore_polarity,
        generic_decay=args.generic_decay,
        min_potential=args.min_potential,
        max_potential=args.max_potential,
        synchronous_decay=args.synchronous_decay,
    )
    return left, right


def frame_image(frame) -> np.ndarray:
    return frame.image


def ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv.cvtColor(image, cv.COLOR_BGRA2BGR)
    return image


def label_image(image: np.ndarray, text: str) -> np.ndarray:
    out = ensure_bgr(image).copy()
    cv.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv.putText(out, text, (8, 20), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)
    return out


def side_by_side(left: np.ndarray, right: np.ndarray, left_label: str = "Left", right_label: str = "Right") -> np.ndarray:
    left_bgr = label_image(left, left_label)
    right_bgr = label_image(right, right_label)
    if left_bgr.shape[0] != right_bgr.shape[0]:
        height = min(left_bgr.shape[0], right_bgr.shape[0])
        left_bgr = cv.resize(left_bgr, (int(left_bgr.shape[1] * height / left_bgr.shape[0]), height))
        right_bgr = cv.resize(right_bgr, (int(right_bgr.shape[1] * height / right_bgr.shape[0]), height))
    return cv.hconcat([left_bgr, right_bgr])


def load_stereo_calibration(
    calibration_file: str | Path,
    left_designation: str = "C0",
    right_designation: str = "C1",
) -> StereoCalibrationPair:
    calibration_set = dv.camera.CalibrationSet.LoadFromFile(str(calibration_file))
    left = calibration_set.getCameraCalibration(left_designation)
    right = calibration_set.getCameraCalibration(right_designation)
    if left is None or right is None:
        cameras = ", ".join(calibration_set.getCameraList())
        raise RuntimeError(
            f"Calibration file does not contain {left_designation}/{right_designation}; available cameras: {cameras}"
        )
    return StereoCalibrationPair(calibration_set=calibration_set, left=left, right=right)


def stereo_geometry_from_file(
    calibration_file: str | Path,
    left_designation: str = "C0",
    right_designation: str = "C1",
):
    calibration = load_stereo_calibration(calibration_file, left_designation, right_designation)
    return calibration, dv.camera.StereoGeometry(calibration.left, calibration.right)


def write_dv_stereo_calibration(
    output_file: str | Path,
    *,
    left_name: str,
    right_name: str,
    image_size: tuple[int, int],
    left_matrix: np.ndarray,
    left_distortion: np.ndarray,
    right_matrix: np.ndarray,
    right_distortion: np.ndarray,
    rotation_left_to_right: np.ndarray,
    translation_left_to_right: np.ndarray,
    essential: np.ndarray,
    fundamental: np.ndarray,
) -> None:
    width, height = image_size
    left_matrix = np.asarray(left_matrix, dtype=np.float32)
    right_matrix = np.asarray(right_matrix, dtype=np.float32)
    left_distortion = np.asarray(left_distortion, dtype=np.float32).reshape(-1)
    right_distortion = np.asarray(right_distortion, dtype=np.float32).reshape(-1)

    right_to_left = np.eye(4, dtype=np.float32)
    rotation_right_to_left = np.asarray(rotation_left_to_right, dtype=np.float32).T
    translation_left_to_right = np.asarray(translation_left_to_right, dtype=np.float32).reshape(3, 1)
    translation_right_to_left = -rotation_right_to_left @ translation_left_to_right
    right_to_left[:3, :3] = rotation_right_to_left
    right_to_left[:3, 3] = translation_right_to_left.reshape(3)

    calibration_set = dv.camera.CalibrationSet()
    camera_metadata = dv.camera.calibrations.CameraCalibration.Metadata()
    stereo_metadata = dv.camera.calibrations.StereoCalibration.Metadata()

    calibration_set.addCameraCalibration(
        dv.camera.calibrations.CameraCalibration(
            left_name,
            "left",
            True,
            (width, height),
            (float(left_matrix[0, 2]), float(left_matrix[1, 2])),
            (float(left_matrix[0, 0]), float(left_matrix[1, 1])),
            [float(value) for value in left_distortion],
            dv.camera.DistortionModel.RADIAL_TANGENTIAL,
            dv.kinematics.Transformationf(),
            camera_metadata,
        )
    )
    calibration_set.addCameraCalibration(
        dv.camera.calibrations.CameraCalibration(
            right_name,
            "right",
            False,
            (width, height),
            (float(right_matrix[0, 2]), float(right_matrix[1, 2])),
            (float(right_matrix[0, 0]), float(right_matrix[1, 1])),
            [float(value) for value in right_distortion],
            dv.camera.DistortionModel.RADIAL_TANGENTIAL,
            dv.kinematics.Transformationf(0, right_to_left),
            camera_metadata,
        )
    )
    calibration_set.addStereoCalibration(
        dv.camera.calibrations.StereoCalibration(
            left_name,
            right_name,
            [float(value) for value in np.asarray(fundamental, dtype=np.float32).reshape(-1)],
            [float(value) for value in np.asarray(essential, dtype=np.float32).reshape(-1)],
            stereo_metadata,
        )
    )
    calibration_set.writeToFile(str(output_file))


def slicer_interval(args) -> timedelta:
    return timedelta(milliseconds=args.interval_ms)
