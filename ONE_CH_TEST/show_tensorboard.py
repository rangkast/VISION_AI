import torch
from torch.utils.tensorboard import SummaryWriter
import sys
import os
print(sys.path)
import cv2
import torch
sys.path.append('/home/rangkast.jeong/workspace/ONE_CH_TEST/ultralytics')
from ultralytics import YOLO

# 1-channel generated model
model_path = './runs/detect/train3/weights/best.pt'

# 3-channel generated model
# model_path = '../VISION_AI/runs/detect/train13/weights/best.pt'

# YOLO 모델 로드 (예시로 yolov5s.pt 사용)
model = YOLO(model_path)  # xxx.pt를 실제 모델 파일 이름으로 바꿉니다.

# 임의의 입력 데이터 생성 (모델의 입력 크기에 맞게 설정)
dummy_input = torch.randn(1, 1, 640, 640)  # 여기서 3은 RGB 채널, 640x640은 입력 크기


# TensorBoard에 모델 그래프를 기록하기 위한 SummaryWriter 생성
writer = SummaryWriter("runs/yolo_model")

# 모델 그래프를 기록 (strict=False 사용)
writer.add_graph(model.model, dummy_input, use_strict_trace=False)

# 기록을 마친 후 Writer를 닫습니다.
writer.close()