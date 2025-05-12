import torch
from thop import profile
from ultralytics import YOLO  # 确保ultralytics库已更新到最新版本


def analyze_model(model_path):
    # 使用YOLO官方加载方式
    model = YOLO(model_path).model  # 获取底层PyTorch模型
    model.eval().to('cuda:0')

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")
    print(f"Non-Trainable Parameters: {(total_params - trainable_params) / 1e6:.2f} M")
    print(f"Memory Usage: {total_params * 4 / 1024 ** 2:.2f} MB (FP32)")

    # FLOPs计算
    try:
        input_tensor = torch.randn(1, 3, 640, 640).to('cuda:0')
        flops, params = profile(
            model,
            inputs=(input_tensor,),
            verbose=False,
            # 排除干扰项
            custom_ops={torch.nn.SiLU: None}  # 针对YOLO的特殊激活函数
        )
        print(f"FLOPs: {flops / 1e9:.2f} GFLOPs @ 640x640")
    except Exception as e:
        print(f"FLOPs计算失败: {str(e)}")


if __name__ == "__main__":
    model_path = r"/runs/WTConv_s_200/weights/best.pt"
    analyze_model(model_path)