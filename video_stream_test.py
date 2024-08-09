import cv2
from yolo_test import *
from image_filter import *

# 카메라 열기
cap = cv2.VideoCapture('/dev/video5')

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('frame size: %d, %d' % (width, height))

image_count = 0
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# 적용할 필터 리스트
filters = [
    apply_grayscale,
    lambda img: apply_darken(img, intensity=0.2)  # 어둡게 하는 필터를 적용
]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # 필터 적용
    frame = apply_custom_filter(frame, filters)
    
    # 3 channel data increasing TEST
    frame_2 = equalizeHistogram(frame)
    # frame_3 = addFilters_on_channel(frame)
    frame_4 = sobel_filter(frame_2)
    
    
    cv2.imshow('Filtered image', frame)
    
    cv2.imshow('TEST image 2', frame_2)
    # cv2.imshow('TEST image 3', frame_3)
    cv2.imshow('TEST image 4', frame_4)
    
    KEY = cv2.waitKey(1) & 0xFF
    image_count += 1
    if KEY == ord('q'):
        break
    elif KEY == ord('s'):
        save_image_and_annotations(image_count, frame, [])

cap.release()
cv2.destroyAllWindows()