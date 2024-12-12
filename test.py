import os

from click.core import batch

from ultralytics import YOLO
import torch

yaml = 'ultralytics/cfg/models/v8/yolov8-irhm.yaml'

model = YOLO(yaml)
# model = YOLO('yolov8n.pt')
# model.info()

if __name__ == "__main__":

    model.to(dtype=torch.float32, device='cuda')
    result = model.train(data='irhm1128.yaml',
                         name='irhm',
                         epochs=50,
                         batch=1)