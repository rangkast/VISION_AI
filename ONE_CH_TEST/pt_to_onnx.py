import torch
from torch.utils.tensorboard import SummaryWriter
import sys
import os
print(sys.path)
import cv2
import torch
sys.path.append('/home/rangkast.jeong/workspace/ONE_CH_TEST/ultralytics')
from ultralytics import YOLO

# 3-channel generated model
model_path = '../VISION_AI/runs/detect/train13/weights/best.pt'

# YOLO 모델 로드 (예시로 yolov5s.pt 사용)
model = YOLO(model_path)  # xxx.pt를 실제 모델 파일 이름으로 바꿉니다.

# success = model.export(format='onnx')

# 2. 더미 입력 텐서 생성 (모델이 예상하는 입력 크기와 동일해야 함)
dummy_input = torch.randn(1, 3, 640, 640)  # 배치 크기 1, 3채널(RGB), 640x640 입력 크기

# 3. ONNX로 모델 변환 및 저장
onnx_file_path = "yolov8.onnx"  # 저장할 ONNX 파일 경로
torch.onnx.export(
    model.model,                # PyTorch 모델
    dummy_input,                # 더미 입력
    onnx_file_path,             # 저장할 ONNX 파일 경로
    export_params=True,         # 모델의 학습된 가중치와 함께 저장
    opset_version=11,           # ONNX 버전
    do_constant_folding=True,   # 상수 폴딩 최적화
    input_names=['input'],      # 입력 이름
    output_names=['output'],    # 출력 이름
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 배치 크기 동적 처리
)