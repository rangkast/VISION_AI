import cv2
import sys
import json
import os
import random
import torch
import mediapipe as mp
from image_filter import *
from ultralytics import YOLO
import platform


from image_filter import *

# YOLOv8 모델 로드
model = YOLO('../yolo_learning/runs/detect/train17/weights/best.pt')
video_path = '/dev/video5'
img_sizes = [480, 640]

# 적용할 필터 리스트
filters = [
    apply_grayscale,
    lambda img: apply_darken(img, intensity=0.2)  # 어둡게 하는 필터를 적용
]


ENABLE = 1
DISABLE = 0
DONE = 'DONE'
NOT_SET = 'NOT_SET'
READ = 0
WRITE = 1
ERROR = -1
SUCCESS = 1

CAM_DELAY = 1

annotation_data = {"images": []}
image_count = 0

# Predefined colors for labels
label_colors = {}

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


def camera_start():
    system = platform.system()
    if system == "Windows":
        cap1 = cv2.VideoCapture(0)
    else:
        cap1 = cv2.VideoCapture(video_path)

    width = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('frame size: %d, %d' % (width, height))

    if not cap1.isOpened():
        sys.exit()

    while True:
        ret1, frame = cap1.read()
        if not ret1:
            break

        # 필터 적용
        frame_gray = apply_custom_filter(frame, filters)
        frame_sobel = sobel_filter(frame_gray)
        
        draw_img = frame.copy()    
      
        results = model.predict(frame_sobel, imgsz=img_sizes, verbose=True)

        ret_array = []
        for result in results:
            for box in result.boxes:
                score = box.conf[0].cpu().item()
                if score < 0.3:
                    continue
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)
                label = f"{int(box.cls[0].cpu().item())} {box.conf[0].cpu().item():.2f}"
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(draw_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)                         
                ret_array.append(([(x1, y1), (x2, y2)], label))

        KEY = cv2.waitKey(1) & 0xFF
        if KEY == 27:  # Esc pressed
            break

        cv2.imshow("video", draw_img)
        cv2.waitKey(CAM_DELAY)

    cap1.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print('available filters')
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # model_path = f"{script_dir}/../data_sets/best.pt"

    img_filter_global = init_filter_models()
    for i, filter in enumerate(img_filter_global):
        print(f"{i}, {filter.get_name()}")

    set_curr_filter(img_filter_global[1])

    camera_start()