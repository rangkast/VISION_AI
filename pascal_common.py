import shutil
import xml.etree.ElementTree as ET
import os
import numpy as np

from image_filter import *

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


def convert_annotation(annotation_path, output_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    with open(output_path, 'w') as out_file:
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls_name = obj.find('name').text
            if int(difficult) == 1 or cls_name not in class_mapping:
                continue
            cls_id = class_mapping[cls_name]
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
            x_center = ((b[0] + b[2]) / 2) / w
            y_center = ((b[1] + b[3]) / 2) / h
            bbox_width = (b[2] - b[0]) / w
            bbox_height = (b[3] - b[1]) / h
            out_file.write(f"{cls_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

def process_and_convert_files(files, img_source_dir, img_target_dir, ann_source_dir, ann_target_dir):
    for f in files:
        img_path = os.path.join(img_source_dir, f + '.jpg')
        shutil.copy(img_path, img_target_dir)

        xml_path = os.path.join(ann_source_dir, f + '.xml')
        txt_path = os.path.join(ann_target_dir, f + '.txt')
        convert_annotation(xml_path, txt_path)

def get_files_from_split(file_path):
    with open(file_path, 'r') as f:
        files = [x.strip().split()[0] for x in f.readlines() if not x.startswith('#')]
    return files

def copy_files(files, source_dir, target_dir, file_ext):
    for f in files:
        shutil.copy(os.path.join(source_dir, f + file_ext), os.path.join(target_dir, f + file_ext))

def dataset_copy_and_convert(base_dir, split_base, split_ratio, do_random=True):
    train_images_dir = os.path.join(split_base, 'images/train')
    val_images_dir = os.path.join(split_base, 'images/val')
    train_labels_dir = os.path.join(split_base, 'labels/train')
    val_labels_dir = os.path.join(split_base, 'labels/val')
    images_dir = os.path.join(base_dir, 'JPEGImages')
    annotations_dir = os.path.join(base_dir, 'Annotations')
    sets_dir = os.path.join(base_dir, 'ImageSets/Main')
        
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    trainval_files = get_files_from_split(os.path.join(sets_dir, 'trainval.txt'))
    if do_random:
        np.random.shuffle(trainval_files)

    split_index = int(len(trainval_files) * split_ratio)
    train_files = trainval_files[:split_index]
    val_files = trainval_files[split_index:]

    copy_files(train_files, images_dir, train_images_dir, '.jpg')
    copy_files(val_files, images_dir, val_images_dir, '.jpg')
    copy_files(train_files, annotations_dir, train_labels_dir, '.xml')
    copy_files(val_files, annotations_dir, val_labels_dir, '.xml')
    
    process_and_convert_files(train_files, images_dir, train_images_dir, annotations_dir, train_labels_dir)
    process_and_convert_files(val_files, images_dir, val_images_dir, annotations_dir, val_labels_dir)

    print(f"Total images (trainval): {len(trainval_files)}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

     
def create_data_yaml(data_path, train_path, val_path):
    # data.yaml 파일을 코드에서 직접 생성
    data_yaml = f"""
    train: {train_path}
    val: {val_path}
    nc: 20
    names: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 
            'train', 'tvmonitor']
    """
    # data.yaml 파일을 저장할 경로
    yaml_path = os.path.join(data_path, "data.yaml")

    # data.yaml 파일 저장
    with open(yaml_path, 'w') as f:
        f.write(data_yaml)

    return yaml_path

