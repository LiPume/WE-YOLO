import os
import cv2
import numpy as np
from ultralytics import YOLO


# 计算 IoU 的函数
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    overlap_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return overlap_area / float(box1_area + box2_area - overlap_area)


# 参数
iou_threshold = 0.5
model_path = r'/root/autodl-tmp/WE-YOLO/runs/neck/z/weights/best.pt'
val_image_dir = r"/root/autodl-tmp/dataset/split_z_0920/images/val"
label_dir = r"/root/autodl-tmp/dataset/split_z_0920/labels/val"
save_dir = r"results/neck/z"
high_accuracy_dir = os.path.join(save_dir, "all")  # 新建文件夹用于保存高准确率图像
results_file = os.path.join(save_dir, "evaluation_metrics.txt")

# 创建文件夹
os.makedirs(high_accuracy_dir, exist_ok=True)

# 加载模型
model = YOLO(model_path)

# 初始化统计变量
true_positive = 0
false_positive = 0
false_negative = 0
total_gt_boxes = 0
total_pred_boxes = 0
correct_pred_boxes = 0

# 获取验证集图片路径
image_paths = [os.path.join(val_image_dir, img) for img in os.listdir(val_image_dir) if
               img.endswith(('.jpg', '.png', '.jpeg'))]

# 遍历每张图片并进行推理
for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image {image_path}")
        continue

    img_h, img_w = img.shape[:2]
    results = model(image_path)
    pred_boxes = results[0].boxes.xyxy.cpu().numpy()
    pred_scores = results[0].boxes.conf.cpu().numpy()
    total_pred_boxes += len(pred_boxes)

    label_file = os.path.join(label_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
    gt_boxes = []
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, x_center, y_center, width, height = map(float, parts[:5])
                    x_center *= img_w
                    y_center *= img_h
                    width *= img_w
                    height *= img_h
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    gt_boxes.append([x1, y1, x2, y2])
    gt_boxes = np.array(gt_boxes)
    total_gt_boxes += len(gt_boxes)

    # 统计逻辑
    matched_gt = set()
    for pred_idx, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        for idx, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            true_positive += 1
            matched_gt.add(best_gt_idx)
            correct_pred_boxes += 1
        else:
            false_positive += 1

    false_negative += len(gt_boxes) - len(matched_gt)

    # 计算当前图像的 Precision
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

    # 如果 Precision 大于或等于 30%，保存图像
    if precision >= 0:
        high_accuracy_path = os.path.join(high_accuracy_dir, os.path.basename(image_path))

        # 绘制预测框和真实框
        for idx, box in enumerate(pred_boxes):
            x1, y1, x2, y2 = map(int, box)
            confidence = pred_scores[idx]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 绘制预测框，红色，线宽2
            label = f'face {confidence:.2f}'  # 标注置信度
            # 检查是否需要调整文本位置
            if y1 - 20 > 0:  # 如果框上方有足够的空间
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # 在预测框上方绘制置信度
            else:  # 如果框上方空间不足，将文本绘制在框下方
                cv2.putText(img, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制真实框，绿色，线宽2
            label = 'original'
            # 检查是否需要调整文本位置
            if y2 + 20 < img_h:  # 如果框下方有足够的空间
                cv2.putText(img, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                            2)  # 在真实框下方标注“original”
            else:  # 如果框下方空间不足，将文本绘制在框上方
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 保存绘制后的图像
        cv2.imwrite(high_accuracy_path, img)

    # 重置统计变量，为下一张图片做准备
    true_positive = 0
    false_positive = 0
    false_negative = 0

# 计算最终的 Precision 和 Recall
precision = correct_pred_boxes / total_pred_boxes if total_pred_boxes > 0 else 0
recall = correct_pred_boxes / total_gt_boxes if total_gt_boxes > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# 输出统计结果
results = (
    f"Total GT Boxes: {total_gt_boxes}\n"
    f"Total Predicted Boxes: {total_pred_boxes}\n"
    f"Correct Predicted Boxes: {correct_pred_boxes}\n"
    f"Precision: {precision:.2f}\n"
    f"Recall: {recall:.2f}\n"
    f"F1 Score: {f1_score:.2f}\n"
)
print(results)

# 保存结果到文件
with open(results_file, "w") as f:
    f.write(results)