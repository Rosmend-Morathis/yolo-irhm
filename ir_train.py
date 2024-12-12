from ultralytics import YOLO
import torch

yaml = 'ultralytics/cfg/models/v8/yolov8-irhm.yaml'

model = YOLO(yaml)
model.to(dtype=torch.float32)
model.train(data='irhm1128.yaml', name='irhm',epochs=200, batch=32, device='1')
metrics = model.val()

