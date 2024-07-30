import cv2
print(cv2.__version__)
import sys

import json
import os
import random
from image_filter import *


CAM_1 = ["/dev/video0", "imgL"]

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

def Rotate(src, degrees):
    if degrees == 90:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 1)
    elif degrees == 180:
        dst = cv2.flip(src, -1)
    elif degrees == 270:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 0)
    else:
        dst = NOT_SET
    return dst

def draw_hand_bounding_box(image, hand_landmarks):
    image_height, image_width, _ = image.shape
    x_min, y_min = image_width, image_height
    x_max, y_max = 0, 0

    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red color

def save_image_and_annotations(frame, rois):
    global image_count
    image_name = f"image_{image_count:04d}.jpg"
    cv2.imwrite(image_name, frame)
    annotations = []
    for roi in rois:
        x_min, y_min, w, h = [int(v) for v in roi['bbox']]
        annotations.append({
            "label": roi['label'],  # Save the label
            "bbox": [x_min, y_min, x_min + w, y_min + h]
        })
    annotation_data["images"].append({
        "file": image_name,
        "annotations": annotations
    })
    image_count += 1
    with open("labels.json", "w") as f:
        json.dump(annotation_data, f, indent=4)

def get_label_color(label):
    if label not in label_colors:
        label_colors[label] = [random.randint(0, 255) for _ in range(3)]
    return label_colors[label]

def camera_start():
    global trackers, rois, tracking

    cap1 = cv2.VideoCapture('/dev/video0')
    width = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('cap1 size: %d, %d' % (width, height))
    if not cap1.isOpened():
        sys.exit()

    while True:
        ret1, frame1 = cap1.read()        
        if not ret1:
            break

        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        filtered_img = add_image_filter(frame1)

        rotate_img = Rotate(filtered_img, 270)

        if tracking:
            success, boxes = trackers.update(rotate_img)
            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                label = rois[i]['label']
                color = get_label_color(label)
                cv2.rectangle(rotate_img, p1, p2, color, 2, 1)
                cv2.putText(rotate_img, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        KEY = cv2.waitKey(1) & 0xFF
        if KEY == 27:  # Esc pressed
            break
        elif KEY == ord('t'):  # 't' pressed, select ROIs
            trackers = cv2.legacy.MultiTracker_create()
            rois = []
            while True:
                bbox = cv2.selectROI("video", rotate_img, fromCenter=False, showCrosshair=True)
                if bbox[2] == 0 or bbox[3] == 0:  # No region selected
                    break
                label = input("Enter label for the selected region: ")
                rois.append({"bbox": bbox, "label": label})
                tracker = cv2.legacy.TrackerCSRT_create()  # Use legacy tracker here
                trackers.add(tracker, rotate_img, bbox)
            tracking = True
        elif KEY == ord('s') and rois:  # 's' pressed, start tracking
            tracking = True
        elif KEY == ord('p') and tracking:  # 'p' pressed, save image and annotations
            save_image_and_annotations(rotate_img, rois)
        elif KEY == ord('c'):  # 'c' pressed, clear all trackers
            trackers = cv2.legacy.MultiTracker_create()
            rois = []
            tracking = False

        cv2.imshow("video", rotate_img)
        cv2.waitKey(CAM_DELAY)

    cap1.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print('available filters')

    img_filter_global = init_filter_models()
    for i, filter in enumerate(img_filter_global):
        print(f"{i}, {filter.get_name()}")
    set_curr_filter(img_filter_global[1])

    camera_start()