import cv2
print(cv2.__version__)
import sys

import json
import os
import random
from image_filter import *
from common_functions import *

CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960

ENABLE = 1
DISABLE = 0
DONE = 'DONE'
NOT_SET = 'NOT_SET'
READ = 0
WRITE = 1
ERROR = -1
SUCCESS = 1

CAM_DELAY = 1

# Define a global variable for the trackers
trackers = cv2.legacy.MultiTracker_create()
rois = []
tracking = False
annotation_data = {"images": []}
image_count = 0

# Predefined colors for labels
label_colors = {}

def camera_start():
    global trackers, rois, tracking
    cap1 = cv2.VideoCapture('/dev/video5')
    width = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('cap1 size: %d, %d' % (width, height))
    if not cap1.isOpened():
        sys.exit()

    while True:
        ret1, frame1 = cap1.read()        
        if not ret1:
            break

        # frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        filtered_img = add_image_filter(frame1)

        rotate_img = Rotate(filtered_img, 0)
        rotate_img = cv2.resize(rotate_img, (960, 540))
        draw_img = rotate_img.copy()

        if tracking:
            success, boxes = trackers.update(draw_img)
            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                label = rois[i]['label']
                color = get_label_color(label)
                cv2.rectangle(draw_img, p1, p2, color, 2, 1)
                cv2.putText(draw_img, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        KEY = cv2.waitKey(1) & 0xFF
        if KEY == 27:  # Esc pressed
            break
        elif KEY == ord('t'):  # 't' pressed, select ROIs
            trackers = cv2.legacy.MultiTracker_create()
            rois = []
            while True:
                bbox = cv2.selectROI("video", draw_img, fromCenter=False, showCrosshair=True)
                if bbox[2] == 0 or bbox[3] == 0:  # No region selected
                    break
                label = input("Enter label for the selected region: ")
                rois.append({"bbox": bbox, "label": label})
                tracker = cv2.legacy.TrackerCSRT_create()  # Use legacy tracker here
                trackers.add(tracker, draw_img, bbox)
            tracking = True
        elif KEY == ord('s') and rois:  # 's' pressed, start tracking
            tracking = True
        elif KEY == ord('p') and tracking:  # 'p' pressed, save image and annotations
            save_image_and_annotations(image_count, rotate_img, rois)
        elif KEY == ord('c'):  # 'c' pressed, clear all trackers
            trackers = cv2.legacy.MultiTracker_create()
            rois = []
            tracking = False

        cv2.imshow("video", draw_img)
        cv2.waitKey(CAM_DELAY)
        image_count += 1

    cap1.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print('available filters')

    img_filter_global = init_filter_models()
    for i, filter in enumerate(img_filter_global):
        print(f"{i}, {filter.get_name()}")
    set_curr_filter(img_filter_global[1])

    camera_start()