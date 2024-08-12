import cv2
import os
import numpy as np
import shutil
import xml.etree.ElementTree as ET
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ultralytics import YOLO

# yolo_v10_learning.py에서 필요한 필터 함수들 임포트
from image_filters import *


# 적용할 필터 리스트
filters = [
    apply_grayscale,
    lambda img: apply_darken(img, intensity=1.0)  # 어둡게 하는 필터를 적용
]

# Pascal VOC 데이터셋의 커스텀 Dataset 클래스 정의
class VOCDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        annotation_path = os.path.join(self.annotation_dir, image_file.replace('.jpg', '.xml'))
        
        image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(image)
        
        boxes = []
        labels = []
        if os.path.exists(annotation_path):
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
                boxes.append([x1, y1, x2, y2])
                labels.append(obj.find('name').text)
        
        return image, boxes, labels


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

     # 결과 모델 저장 경로
     # result_model_dir = os.path.join(script_dir, "result_model")
     # os.makedirs(result_model_dir, exist_ok=True)
     # model_save_path = os.path.join(result_model_dir, "best.pt")
     # model.save(model_save_path)    

#     os.system(f"python train.py --img {img_size[0]} --batch {batch_size} --epochs {epochs} --data {data_yaml} --weights {model_path} --lr {learning_rate}")

# 필터를 적용한 이미지를 저장하는 함수 (train, val, test로 분할)
def save_filtered_images(data_path, output_path, apply_filter=None, train_ratio=0.7, val_ratio=0.2):
     images_dir = os.path.join(data_path, 'JPEGImages')
     annotations_dir = os.path.join(data_path, 'Annotations')

     image_files = sorted(os.listdir(images_dir))
     os.makedirs(output_path, exist_ok=True)

     # 데이터셋을 train, val, test로 분할
     train_files, temp_files = train_test_split(image_files, test_size=(1 - train_ratio))
     val_files, test_files = train_test_split(temp_files, test_size=0.5)  # temp_files을 50%로 분할하여 val과 test로 사용

     # 각 디렉토리 설정
     dirs = {
          'train': train_files,
          'val': val_files,
          'test': test_files
     }

     for split, files in dirs.items():
          split_image_dir = os.path.join(output_path, split, 'images')
          split_annotation_dir = os.path.join(output_path, split, 'annotations')
          os.makedirs(split_image_dir, exist_ok=True)
          os.makedirs(split_annotation_dir, exist_ok=True)

          for image_file in files:
               image_path = os.path.join(images_dir, image_file)
               annotation_path = os.path.join(annotations_dir, image_file.replace('.jpg', '.xml'))

               image = cv2.imread(image_path)

               # 필터 적용
               image = apply_custom_filter(image, filters)
               
               # channel test
               if apply_filter is not None:
                    image = apply_filter(image)       

               cv2.imwrite(os.path.join(split_image_dir, image_file), image)

               # 라벨 파일 복사
               if os.path.exists(annotation_path):
                    shutil.copy(annotation_path, split_annotation_dir)

# 데이터셋 준비 및 학습 파이프라인
def main():
     data_path = os.path.abspath('../pascal_voc/VOCdevkit/VOC2012')  # Pascal VOC 데이터셋 경로를 절대 경로로 설정
     output_path = os.path.abspath('../pascal_voc/filtered_dataset')  # 필터링된 데이터셋 경로를 절대 경로로 설정


     # 필터를 적용하여 이미지 저장 및 데이터셋 분할
     # save_filtered_images(data_path, output_path, apply_filter=None)

     # Train, Validation 데이터셋 경로 설정
     train_dir = os.path.join(output_path, 'train')
     val_dir = os.path.join(output_path, 'val')

     # 학습을 위한 yaml 파일 생성 (Pascal VOC용)
     data_yaml = """
     train: {}
     val: {}
     nc: 20
     names: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 
               'train', 'tvmonitor']
     """.format(os.path.join(train_dir, 'images'), os.path.join(val_dir, 'images'))

     with open(os.path.join(output_path, "data.yaml"), 'w') as f:
          f.write(data_yaml)

     # CUDA 설정
     torch.cuda.empty_cache()
     os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
     if torch.cuda.is_available():
          print("Using GPU")
     else:
          print("Using CPU")

     # YOLO 모델 학습
     train_yolo_model(os.path.join(output_path, "data.yaml"), model_path='yolov10n.pt', epochs=50, batch_size=2, learning_rate=0.001, img_size=(640, 640))

if __name__ == '__main__':
     main()