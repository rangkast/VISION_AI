import random
import os
import cv2
from ultralytics import YOLO

from wandb_common import *
from pascal_common import *

# API 키 파일 경로
api_key_file = "../wandb_api.txt"
# Pascal VOC 2007 데이터셋 경로 설정
base_dir = os.path.abspath('../pascal_voc/VOCdevkit/VOC2007')
# Model path
model_path = './runs/detect/train13/weights/best.pt'
split_base = os.path.abspath('../pascal_voc/yolo_dataset_2007')


# 적용할 필터 리스트
filters = [
    apply_grayscale,
    lambda img: apply_darken(img, intensity=0.2),  # 어둡게 하는 필터를 적용
    # lambda img: sobel_filter(img),
    lambda img: equalizeHistogram(img),
]

def validate_model(model_path, data_yaml):
    # WandB 초기화
    wandb.init(project="YOLOv10 Analysis")

    # 모델 로드
    model = YOLO(model_path)

    # 모델 성능 검증
    results = model.val(
        data=data_yaml,  # 데이터셋을 정의한 YAML 파일 경로
        imgsz=(640, 640),  # 이미지 크기
        conf=0.25,  # confidence threshold (신뢰도 임계값)
        iou=0.5  # IoU threshold (IoU 임계값)
    )

    # results 객체의 구조를 확인하는 코드
    print(results.results_dict)  # 결과를 딕셔너리로 출력

    # 성능 지표 로그 기록
    metrics = results.results_dict
    wandb.log({
        "test_mAP_0.5": metrics['metrics/mAP50(B)'],
        "test_mAP_0.5:0.95": metrics['metrics/mAP50-95(B)'],
        "test_precision": metrics['metrics/precision(B)'],
        "test_recall": metrics['metrics/recall(B)'],
        "fitness": metrics['fitness']
    })

    return results


def show_images_with_boxes(model, image_dir, num_images_to_show=5):
    # 이미지 디렉토리에서 모든 이미지 파일을 가져오기
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    # 선택할 이미지의 인덱스를 무작위로 섞기
    indices = random.sample(range(len(image_files)), num_images_to_show)

    for i in indices:
        img_path = image_files[i]
        img = cv2.imread(img_path)

        # 모델을 사용해 예측
        results = model.predict(source=img, imgsz=640)

        draw_img = img.copy()

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
    # login_wandb(api_key_file)

    dataset_copy_and_convert(base_dir, split_base, split_ratio=0.5)

    yaml_path = create_data_yaml(split_base,
                                 os.path.join(split_base, 'images/train'),
                                 os.path.join(split_base, 'images/val'))
  
    apply_filters_to_images(os.path.join(split_base, 'images/train'), filters)
    apply_filters_to_images(os.path.join(split_base, 'images/val'), filters)
    
    validate_model(model_path, yaml_path)
    wandb.finish()
    
    show_images_with_boxes(YOLO(model_path), os.path.join(split_base, 'images/val'), num_images_to_show=20)

if __name__ == '__main__':
     main()