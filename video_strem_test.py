import cv2
from common_functions import *

cap = cv2.VideoCapture('/dev/video5')

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('frame size: %d, %d' % (width, height))

image_count = 0
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    cv2.imshow('Video Stream', frame)
    KEY = cv2.waitKey(1) & 0xFF
    image_count += 1
    if KEY == ord('q'):
        break
    elif KEY == ord('s'):
        save_image_and_annotations(image_count, frame, [])

cap.release()
cv2.destroyAllWindows()