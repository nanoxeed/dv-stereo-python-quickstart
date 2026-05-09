from __future__ import annotations

import argparse

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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview accumulated stereo event images.")
    add_camera_selection_args(parser)
    add_slicer_args(parser)
    add_accumulator_args(parser)
    parser.add_argument("--window-name", default="Accumulated stereo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair = open_stereo_cameras(args.left, args.right)
    left_accumulator, right_accumulator = create_stereo_accumulators(pair, args)

    cv.namedWindow(args.window_name, cv.WINDOW_NORMAL)
    slicer = dv.StereoEventStreamSlicer()
    keep_running = True

    def preview(left_events, right_events) -> None:
        nonlocal keep_running
        left_accumulator.accept(left_events)
        right_accumulator.accept(right_events)
        left = frame_image(left_accumulator.generateFrame())
        right = frame_image(right_accumulator.generateFrame())
        cv.imshow(args.window_name, side_by_side(left, right, pair.left_name, pair.right_name))
        if cv.waitKey(2) == 27:
            keep_running = False

    slicer.doEveryTimeInterval(slicer_interval(args), preview)

    while keep_running and cameras_are_running(pair):
        slicer.accept(*next_stereo_events(pair))

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
