import cv2

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
          dst = src
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

def save_image_and_annotations(image_count, frame, rois):
     image_name = f"image_{image_count:04d}.jpg"
     cv2.imwrite(image_name, frame)
    
     if len(rois) > 0:
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
