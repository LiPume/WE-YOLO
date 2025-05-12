import argparse
from ultralytics import YOLO


def parse_args():
    parse = argparse.ArgumentParser(description='Data Postprocess')
    parse.add_argument('--model', type=str, default=None, help='load the model')
    parse.add_argument('--data_dir', type=str, default=None, help='the dir to data')
    args = parse.parse_args()
    return args


def main():
    args = parse_args()
    model = YOLO(r'E:\AIpractice\detect\yolov11\ultralytics-main\ultralytics\cfg\models\11\yolo11WTEMA.yaml')

    model.train(
        data='/root/autodl-tmp/Fracture_Detection_Improved_YOLOv8-main/ultralytics/cfg/datasets/boons.yaml',
        cache=False,
        imgsz=640,
        epochs=100,
        single_cls=True,
        batch=4,
        close_mosaic=10,
        workers=0,
        device='0',
        optimizer='SGD',
        resume=True,
        amp=False,
        project='runs/train',
        name='exp'
    )


if __name__ == '__main__':
    main()
