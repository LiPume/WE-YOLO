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
model_path = r"/runs/WTConv+EMA_z_200/weights/best.pt"
val_image_dir = r"/dataset/split_z_0920/images/val"
label_dir = r"/dataset/split_z_0920/labels/val"
save_dir = r"E:\AIpractice\detect\yolov11\ultralytics-main\results\az"
all_images_dir = os.path.join(save_dir, "全部")
multidet_dir = os.path.join(save_dir, "多检")
missed_dir = os.path.join(save_dir, "漏检")
results_file = os.path.join(save_dir, "evaluation_metrics.txt")

# 创建文件夹
os.makedirs(all_images_dir, exist_ok=True)
os.makedirs(multidet_dir, exist_ok=True)
os.makedirs(missed_dir, exist_ok=True)

# 加载模型
model = YOLO(model_path)

# 初始化统计变量
true_positive = 0
false_positive = 0
false_negative = 0
total_gt_boxes = 0
total_pred_boxes = 0
correct_pred_boxes = 0
total_multidet = 0
total_missed = 0

# 获取验证集图片路径
image_paths = [os.path.join(val_image_dir, img) for img in os.listdir(val_image_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]

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

    # 生成多检和漏检逻辑
    matched_pred = set()
    multidet_boxes = []
    missed_boxes = []

    # 多检框逻辑
    for pred_idx, pred_box in enumerate(pred_boxes):
        if pred_idx not in matched_pred:
            best_iou = 0
            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
            if best_iou < iou_threshold:
                multidet_boxes.append(pred_box)
    total_multidet += len(multidet_boxes)

    # 漏检框逻辑
    for gt_idx, gt_box in enumerate(gt_boxes):
        if gt_idx not in matched_gt:
            missed_boxes.append(gt_box)
    total_missed += len(missed_boxes)

    # 绘制所有图像
    for idx, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = map(int, box)
        confidence = pred_scores[idx]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        label = f'fracture {confidence:.2f}'
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    for box in gt_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(img, 'original', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # 保存全部图像
    all_path = os.path.join(all_images_dir, os.path.basename(image_path))
    cv2.imwrite(all_path, img)

    # 保存多检图像
    if len(multidet_boxes) > 0:
        multidet_path = os.path.join(multidet_dir, os.path.basename(image_path))
        cv2.imwrite(multidet_path, img)

    # 保存漏检图像
    if len(missed_boxes) > 0:
        missed_path = os.path.join(missed_dir, os.path.basename(image_path))
        cv2.imwrite(missed_path, img)

# 计算 Precision 和 Recall
precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# 输出统计结果
results = (
    f"Total GT Boxes: {total_gt_boxes}\n"
    f"Total Predicted Boxes: {total_pred_boxes}\n"
    f"Correct Predicted Boxes: {correct_pred_boxes}\n"
    f"Precision: {precision:.2f}\n"
    f"Recall: {recall:.2f}\n"
    f"F1 Score: {f1_score:.2f}\n"
    f"True Positives: {true_positive}\n"
    f"False Positives: {false_positive}\n"
    f"False Negatives: {false_negative}\n"
    f"Total Multidet Boxes: {total_multidet}\n"
    f"Total Missed Boxes: {total_missed}\n"
)
print(results)

# 保存结果到文件
with open(results_file, "w") as f:
    f.write(results)
