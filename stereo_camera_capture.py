import argparse

import cv2 as cv
import dv_processing as dv

from stereo_common import (
    add_camera_selection_args,
    add_slicer_args,
    cameras_are_running,
    next_stereo_events,
    open_stereo_cameras,
    slicer_interval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a raw stereo event-stream preview from synchronized iniVation cameras."
    )
    add_camera_selection_args(parser)
    add_slicer_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair = open_stereo_cameras(args.left, args.right)

    left_visualizer = dv.visualization.EventVisualizer(pair.left.getEventResolution())
    right_visualizer = dv.visualization.EventVisualizer(pair.right.getEventResolution())

    cv.namedWindow("Left", cv.WINDOW_NORMAL)
    cv.namedWindow("Right", cv.WINDOW_NORMAL)

    slicer = dv.StereoEventStreamSlicer()
    keep_running = True

    def preview(left_events, right_events) -> None:
        nonlocal keep_running
        cv.imshow("Left", left_visualizer.generateImage(left_events))
        cv.imshow("Right", right_visualizer.generateImage(right_events))
        if cv.waitKey(2) == 27:
            keep_running = False

    slicer.doEveryTimeInterval(slicer_interval(args), preview)

    while keep_running and cameras_are_running(pair):
        slicer.accept(*next_stereo_events(pair))

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
