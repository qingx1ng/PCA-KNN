import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from config.settings import Config

def plot_confusion_matrix(y_true, y_pred, classes=None):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if classes is not None:
        plt.xticks(np.arange(len(classes))+0.5, classes)
        plt.yticks(np.arange(len(classes))+0.5, classes)
    
    # 保存图像
    os.makedirs(Config.PLOT_SAVE_PATH, exist_ok=True)
    plt.savefig(os.path.join(Config.PLOT_SAVE_PATH, 'confusion_matrix.png'))
    plt.close()

def save_metrics(metrics, filename='baseline_metrics.txt'):
    """
    保存评估指标到文件
    """
    os.makedirs('results/metrics', exist_ok=True)
    with open(os.path.join('results/metrics', filename), 'w') as f:
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(metrics['report'])