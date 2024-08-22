import sys
import os
print(sys.path)
import cv2
import random
sys.path.append('/home/rangkast.jeong/workspace/ONE_CH_TEST/ultralytics')
from ultralytics import YOLO
import torch
from image_filter import *

model_path = './runs/detect/train3/weights/best.pt'
split_base = os.path.abspath('./pascal_voc/yolo_dataset_2012')


class_mapping = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3,
    'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7,
    'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
    'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15,
    'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
}

# 클래스 ID를 이름으로 변환하는 함수
def get_class_name(class_id):
    for name, id_ in class_mapping.items():
        if id_ == class_id:
            return name
    return "Unknown"


# 적용할 필터 리스트
filters = [
    lambda img: apply_grayscale(img),
    lambda img: apply_darken(img, intensity=0.2),  # 어둡게 하는 필터를 적용
    # lambda img: sobel_filter(img),
    lambda img: equalizeHistogram(img),
]

def show_images_with_boxes(model, image_dir, num_images_to_show=5):
    # 이미지 디렉토리에서 모든 이미지 파일을 가져오기
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    # 선택할 이미지의 인덱스를 무작위로 섞기
    indices = random.sample(range(len(image_files)), num_images_to_show)

    for i in indices:
        img_path = image_files[i]
        img = cv2.imread(img_path)
        
        # 필터 적용
        filtered_image = apply_custom_filter(img, filters)


        # 형태 확인
        print(f"Original image shape: {img.shape}")
        print(f"Filtered image shape: {filtered_image.shape}")


        # 배치 차원 추가 및 채널 순서 변경
        if len(filtered_image.shape) == 3 and filtered_image.shape[2] == 1:
            # 이미지가 이미 1채널일 경우 (H, W, 1)
            filtered_image = np.transpose(filtered_image, (2, 0, 1))  # (H, W, 1) -> (1, H, W)
        elif len(filtered_image.shape) == 2:
            # 이미지가 이미 2D일 경우 (H, W)
            filtered_image = np.expand_dims(filtered_image, axis=0)  # (H, W) -> (1, H, W)
        else:
            # 3채널 이미지인 경우 (H, W, 3)
            filtered_image = np.transpose(filtered_image, (2, 0, 1))  # (H, W, 3) -> (3, H, W)

        filtered_image = np.expand_dims(filtered_image, axis=0)  # 배치 차원 추가 (C, H, W) -> (1, C, H, W)

        print(f"Final filtered image shape for model: {filtered_image.shape}")

        # 모델을 사용해 예측 (이미지가 1채널인지 확인 후 예측)
        filtered_image_tensor = torch.from_numpy(filtered_image).float()
        results = model.predict(source=filtered_image, imgsz=(640, 640))

        draw_img = filtered_image.copy()

        # 실제 라벨과 예측 결과를 모두 그리기
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)
                class_id = int(box.cls[0].cpu().item())
                class_name = get_class_name(class_id)
                conf = box.conf[0].cpu().item()

                label = f"{class_name} ({conf:.2f})"
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(draw_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # 이미지 표시
        cv2.imshow("Image", draw_img)
        key = cv2.waitKey(0) & 0xFF  # 키 입력 대기
        if key == ord('n'):  # 'n' 키를 누르면 다음 이미지로 넘어감
            continue
        elif key == 27:  # 'Esc' 키를 누르면 종료
            break

    cv2.destroyAllWindows()

def main():    
    show_images_with_boxes(YOLO(model_path), os.path.join(split_base, 'images/val'), num_images_to_show=20)

if __name__ == '__main__':
     main()