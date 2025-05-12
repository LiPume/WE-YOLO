import warnings
import os
from ultralytics import YOLO
import matplotlib
matplotlib.use('TkAgg')  # 设置为交互式后端
import matplotlib.pyplot as plt
if __name__ == '__main__':
    # 忽略特定的警告
    warnings.filterwarnings('ignore', category=UserWarning)

    # 确保保存结果的目录存在
    if not os.path.exists('../detect/feature'):
        os.makedirs('../detect/feature')

    # 加载模型
    model = YOLO(r"/runs/WTConv_s_200/weights/best.pt")

    # 预测
    model.predict(
        source=r'E:\AIpractice\detect\yolov11\ultralytics-main\dataset\split_s_1114\images\val\5s_146.jpg',
        imgsz=640,
        project='detect/feature2',
        name='WTYOLO',
        save=True,
        visualize=True,
        conf=0.25,  # 置信度阈值
        iou=0.45,  # IoU 阈值
        agnostic_nms=False,  # 是否在 NMS 中忽略类别信息
        line_width=3,  # 边界框的线宽
        show_conf=True,  # 显示预测置信度
        show_labels=True,  # 显示预测标签
        save_txt=True,  # 保存预测结果为 .txt 文件
        save_crop=True  # 保存裁剪后的图片
    )