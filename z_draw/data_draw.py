import matplotlib
matplotlib.use('Agg')  # 确保使用合适的后端

import matplotlib.pyplot as plt
import numpy as np

# 数据
x_labels = ['Axial Plane', 'Coronal Plane', 'Sagittal Plane']
all_data = [51845, 82689, 61474]
train_data = [41476, 66151, 49179]
test_data = [10369, 16538, 12295]  # 假设的测试集数据

# 设置柱状图的位置和宽度
x = np.arange(len(x_labels))
width = 0.2

fig, ax = plt.subplots()

# 绘制柱状图
bars1 = ax.bar(x - width, all_data, width, label='All Data', color='skyblue')
bars2 = ax.bar(x, train_data, width, label='Train Data', color='orange')
bars3 = ax.bar(x + width, test_data, width, label='Test Data', color='green')

# 添加标题和标签
ax.set_title('Data Distribution')
ax.set_xlabel('Planes')
ax.set_ylabel('Number')
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.legend()

# 在柱状图上方标注数据
def add_labels(bars):
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 500, int(yval), ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# 保存图像
plt.savefig('data_distribution.png')