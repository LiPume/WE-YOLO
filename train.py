# -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch

torch.cuda.empty_cache()  # 清空未使用的显存缓存
torch.cuda.ipc_collect()  # 收集共享内存

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 初始化 YOLO 模型
model = YOLO('yolo11n.yaml')  # 使用 YOLO 模型配置

# 开始训练
model.train(
    data=r'ultralytics\cfg\datasets\coco128.yaml',  # 数据文件路径
    cache=False,  # 不缓存数据
    imgsz=640,  # 输入图像大小设置为 640x640
    epochs=200,  # 训练轮次
    batch=4,  # 批次大小
    close_mosaic=10,  # 在第10轮关闭 Mosaic 数据增强
    workers=0,  # 数据加载线程数
    device='0',  # 使用 GPU 0
    optimizer='SGD',  # 使用 SGD 优化器
    resume=True,
    amp=True,  # 使用混合精度训练
    project='runs/firstry',  # 输出路径
    name='coco128'  # 项目名称

)
