#Model_Evaluate.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 混淆矩阵数据（假设的真实数据）
cm = np.array([[1081, 12], [6, 901]])  # 你的混淆矩阵
labels = ['Literal', 'Metaphor']  # 标签对应

# 分类指标数据（假设的评估结果）
accuracy = 0.9910
precision = 0.9869
recall = 0.9934
f1 = 0.9901

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
scores = [accuracy, precision, recall, f1]

# 可视化混淆矩阵的函数
def plot_confusion_matrix(ax, cm, labels):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')

# 可视化分类指标条形图的函数
def plot_metrics(ax, metrics, scores):
    ax.bar(metrics, scores, color=['blue', 'orange', 'green', 'red'])
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Model Evaluation Metrics')
    ax.set_ylim(0, 1)
    for i, v in enumerate(scores):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom')

# 绘制图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 创建左侧的混淆矩阵图
plot_confusion_matrix(ax1, cm, labels)

# 创建右侧的分类指标条形图
plot_metrics(ax2, metrics, scores)

plt.tight_layout()
plt.show()
