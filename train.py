import argparse
from utils.data_loader import load_orl_dataset, preprocess_data
from models.base_model import PCA_KNN_Model
from utils.evaluator import save_metrics, plot_confusion_matrix
from utils.visualizer import visualize_eigenfaces, plot_reconstruction, plot_pca_variance, plot_pca_scatter, plot_knn_k_selection
from config.settings import Config

def main():
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--pca', type=int, default=Config.PCA_COMPONENTS, 
                       help='Number of PCA components')
    parser.add_argument('--knn', type=int, default=Config.KNN_NEIGHBORS,
                       help='Number of neighbors for KNN')
    args = parser.parse_args()
    
    # 加载和预处理数据
    X, y = load_orl_dataset()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # 训练模型
    model = PCA_KNN_Model(n_components=args.pca, n_neighbors=args.knn)
    model.fit(X_train, y_train)
    
    # 评估模型
    metrics = model.evaluate(X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    print(metrics['report'])
    
    # 保存结果
    save_metrics(metrics)
    plot_confusion_matrix(y_test, model.predict(X_test))

    # 可视化
    visualize_eigenfaces(model.pca)
    plot_pca_variance(X_train, threshold=0.95)
    plot_pca_scatter(X_train, y_train, n_components=2)
    X_pca = model.pca.fit_transform(X_train)
    plot_knn_k_selection(X_pca, y_train, k_range=range(1, 15))

    # 保存模型
    model.save_model("results/baseline_model.pkl")

if __name__ == "__main__":
    main()