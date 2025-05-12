import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'..\ultralytics\cfg\models\11\yolo11WTEMA.yaml')
    # 开始训练
    model.train(
        data=r'..\ultralytics\cfg\datasets\boons.yaml',
        cache=False,
        imgsz=640,
        epochs=100,
        single_cls=True,
        batch=4,
        close_mosaic=10,
        workers=0,
        device='0',
        optimizer='SGD',  # 使用SGD优化器
        resume=True,  # 如过想续训就设置last.pt的地址
        amp=False,  # 如果出现训练损失为NaN，可以关闭amp
        project='runs/train',
        name='exp'  # 给新项目一个名称，以便与之前的模型区分
    )
