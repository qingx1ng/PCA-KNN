import argparse
from models.base_model import PCA_KNN_Model
from utils.data_loader import load_orl_dataset, preprocess_data
from utils.evaluator import save_metrics, plot_confusion_matrix

def evaluate_model(model_path, output_prefix='baseline'):
    # 加载模型
    model = PCA_KNN_Model.load_model(model_path)
    
    # 加载数据
    X, y = load_orl_dataset()
    _, X_test, _, y_test = preprocess_data(X, y)
    
    # 评估
    metrics = model.evaluate(X_test, y_test)
    
    # 保存结果
    save_metrics(metrics, f"{output_prefix}_metrics.txt")
    plot_confusion_matrix(y_test, model.predict(X_test))
    
    print(f"Evaluation completed. Results saved with prefix: {output_prefix}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="results/baseline_model.pkl",
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default="baseline",
                       help='Prefix for output files')
    args = parser.parse_args()
    
    evaluate_model(args.model, args.output)