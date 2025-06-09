import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n.yaml')

    model.train(
        data=r'/root/autodl-tmp/WE-YOLO/ultralytics/cfg/datasets/boons_z.yaml',
        cache=False,
        imgsz=640,
        epochs=200,
        single_cls=True,
        batch=16,
        close_mosaic=10,
        workers=0,
        device='0',
        optimizer='SGD', 
        resume=True, 
        amp=False, 
        project='runs/WTConv',
        name='1z' 
    )
