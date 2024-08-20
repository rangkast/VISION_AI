import os
import torch
from ultralytics import YOLO

from pascal_common import *

script_dir = os.path.dirname(os.path.realpath(__file__))

base_dir = os.path.abspath('../pascal_voc/VOCdevkit/VOC2012')
split_base = os.path.abspath('../pascal_voc/yolo_dataset_2012')

# 적용할 필터 리스트
filters = [
    apply_grayscale,
    lambda img: apply_darken(img, intensity=0.2),  # 어둡게 하는 필터를 적용
    # lambda img: sobel_filter(img),
    lambda img: equalizeHistogram(img),
]

def train_yolo_model(data_yaml, model_path, epochs=50, batch_size=16, learning_rate=0.001, img_size=(640, 640)):
     model = YOLO(model_path)

     print(f"Training YOLO model with the following configuration:")
     print(f"Data: {data_yaml}")
     print(f"Model: {model_path}")
     print(f"Epochs: {epochs}")
     print(f"Batch size: {batch_size}")
     print(f"Learning rate: {learning_rate}")
     print(f"Image size: {img_size}")

     model.train(
          data=data_yaml,
          epochs=epochs,
          batch=batch_size,
          imgsz=img_size,
          lr0=learning_rate,
          optimizer='AdamW',
          conf=0.1  # Confidence threshold를 낮게 설정하여 더 많은 예측을 수집
     )

if __name__ == '__main__':
    dataset_copy_and_convert(base_dir, split_base, split_ratio=0.9)

    yaml_path = create_data_yaml(split_base,
                                 os.path.join(split_base, 'images/train'),
                                 os.path.join(split_base, 'images/val'))
    # CUDA 설정
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")

    apply_filters_to_images(os.path.join(split_base, 'images/train'), filters)
    apply_filters_to_images(os.path.join(split_base, 'images/val'), filters)

    # YOLO 모델 학습
    train_yolo_model(os.path.join(split_base, "data.yaml"), model_path='yolov10s.pt', epochs=50, batch_size=4, learning_rate=0.001, img_size=(640, 640))


