import cv2

cap = cv2.VideoCapture('/dev/video5')

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    cv2.imshow('Video Stream', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()