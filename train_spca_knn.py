import os
import argparse
import numpy as np
from models.spca_knn_model import SPCA_KNN_Model
from utils.data_loader import load_orl_dataset, preprocess_data
from utils.evaluator import save_metrics, plot_confusion_matrix
from utils.visualizer import (
    visualize_eigenfaces,
    plot_pca_variance,
    plot_pca_scatter,
    plot_knn_k_selection
)
from config.settings import Config

def main():
    # 参数解析器
    parser = argparse.ArgumentParser(description='SPCA+KNN人脸分类训练')
    parser.add_argument('--components', type=int, default=Config.SPCA_COMPONENTS,
                      help=f'稀疏PCA主成分数 (默认: {Config.SPCA_COMPONENTS})')
    parser.add_argument('--alpha', type=float, default=Config.SPCA_ALPHA,
                      help=f'L1正则化系数 (默认: {Config.SPCA_ALPHA})')
    parser.add_argument('--max_iter', type=int, default=Config.SPCA_MAX_ITER,
                      help=f'最大迭代次数 (默认: {Config.SPCA_MAX_ITER})')
    parser.add_argument('--knn', type=int, default=Config.KNN_NEIGHBORS,
                      help=f'KNN邻居数 (默认: {Config.KNN_NEIGHBORS})')
    args = parser.parse_args()

    # 数据准备
    X, y = load_orl_dataset()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # 模型训练
    model = SPCA_KNN_Model(
        n_components=args.components,
        alpha=args.alpha,
        max_iter=args.max_iter,
        n_neighbors=args.knn
    )
    model.fit(X_train, y_train)

    # 模型评估
    metrics = model.evaluate(X_test, y_test)
    print(f"测试准确率: {metrics['accuracy']:.4f}")
    print("\n分类报告:")
    print(metrics['report'])

    # 结果保存
    save_metrics(metrics, filename='spca_knn_metrics.txt')
    plot_confusion_matrix(y_test, model.predict(X_test))

    # 可视化分析
    visualize_eigenfaces(model.pipeline.named_steps['spca'], 
                       image_shape=(112, 92))
    plot_pca_variance(X_train, threshold=Config.VARIANCE_THRESHOLD)
    
    X_spca = model.pipeline.named_steps['spca'].transform(X_train)
    plot_knn_k_selection(X_spca, y_train, k_range=range(1, 15))

    # 保存模型
    model.save_model("results/spca_knn_model.pkl")

if __name__ == "__main__":
    main()