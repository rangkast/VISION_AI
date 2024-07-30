import cv2
import sys
import json
import os
import random
import torch
import mediapipe as mp
from image_filter import *
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('best.pt')

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

def get_label_color(label):
    if label not in label_colors:
        label_colors[label] = [random.randint(0, 255) for _ in range(3)]
    return label_colors[label]

def save_image_and_annotations(frame, annotations):
    global image_count
    image_name = f"image_{image_count:04d}.jpg"
    cv2.imwrite(image_name, frame)
    annotation_data["images"].append({
        "file": image_name,
        "annotations": annotations
    })
    image_count += 1
    with open("labels.json", "w") as f:
        json.dump(annotation_data, f, indent=4)

def camera_start():
     cap1 = cv2.VideoCapture('/dev/video0')
     width = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
     height = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
     print('cap1 size: %d, %d' % (width, height))
     mp_hands = mp.solutions.hands
     mp_drawing = mp.solutions.drawing_utils
     if not cap1.isOpened():
          sys.exit()
     with mp_hands.Hands(
          static_image_mode=False,
          max_num_hands=2,
          min_detection_confidence=0.5,
          min_tracking_confidence=0.5) as hands:
          while True:
               ret1, frame1 = cap1.read()
               if not ret1:
                    break

               frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

               filtered_img = add_image_filter(frame1)

               rotate_img = Rotate(filtered_img, 270)
               rotate_img_color = cv2.cvtColor(rotate_img, cv2.COLOR_GRAY2BGR)

               # YOLO 객체 탐지
               model = YOLO("./best.pt")
               results = model.predict(rotate_img_color, imgsz=[960, 544], verbose=True)
               ret_array = []
               for result in results:
                    for box in result.boxes:
                         score = box.conf[0].cpu().item()
                         label = f"{int(box.cls[0].cpu().item())}"
                         if score < 0.3:
                              continue
                         coords = box.xyxy[0].cpu().numpy()
                         x1, y1, x2, y2 = map(int, coords)
                         ret_array.append(([(x1, y1), (x2, y2)], label))

               # MediaPipe를 이용한 손 감지
               # Convert the BGR image to RGB
               rotate_img_rgb = cv2.cvtColor(rotate_img, cv2.COLOR_BGR2RGB)
               # Process the image and detect hands
               result = hands.process(rotate_img_rgb)
               # Draw hand annotations and bounding boxes on the image            
               if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                         mp_drawing.draw_landmarks(
                              rotate_img_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                         draw_hand_bounding_box(rotate_img_rgb, hand_landmarks)


               KEY = cv2.waitKey(1) & 0xFF
               if KEY == 27:  # Esc pressed
                    break
               cv2.imshow("video", rotate_img_color)
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