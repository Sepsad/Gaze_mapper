import copy
import time

import cv2 as cv
from pupil_apriltags import Detector

# get data from camera
from pupil_labs.realtime_api.simple import discover_one_device

from utils import *

at_detector = Detector(
    families='tag36h11',
    nthreads=1,
    quad_decimate=2.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0,
)


def main():
    # cap = cv.VideoCapture(0)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    # Look for devices. Connects as soon as it has found the first device.
    print("Looking for a device")
    device = discover_one_device(max_search_duration_seconds=10)
    if device is None:
        print("No device found.")
        raise SystemExit(-1)

    print(f"Connecting to {device}...")

    while True:
        start_time = time.time()

        # ret, image = cap.read()
        # print(image)
        
        frame, gaze = device.receive_matched_scene_video_frame_and_gaze()
        image = frame.bgr_pixels

        # main part of code
        img_copy = copy.deepcopy(image)

        img_copy = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
        tags = at_detector.detect(
            img_copy,
            estimate_tag_pose=False,
            camera_params=None,
            tag_size=None,
        )
        if(len(tags) != 4):
            continue

        mapped_image, mapped_gaze = perspective_mapper(
            tags, image, gaze, maxWidth=300, maxHeight=200)

        print(mapped_gaze)
        cv.circle(
            mapped_image,
            (int(mapped_gaze[0][0][0]), int(mapped_gaze[0][0][1])),
            # (50, 50),
            radius=30,
            color=(0, 0, 255),
            thickness=10,
        )

        cv.imshow('Mapped', mapped_image)

        elapsed_time = time.time() - start_time

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

    #     cv.imshow('AprilTag Detect Demo', image)

    # cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
