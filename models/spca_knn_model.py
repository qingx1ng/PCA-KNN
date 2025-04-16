import numpy as np
import joblib
from sklearn.decomposition import SparsePCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from config.settings import Config

class SPCA_KNN_Model:
    def __init__(self, 
                 n_components=None, 
                 alpha=None,
                 max_iter=None,
                 n_neighbors=None,
                 metric=None):
        
        # 参数初始化
        self.n_components = n_components or Config.SPCA_COMPONENTS
        self.alpha = alpha or Config.SPCA_ALPHA
        self.max_iter = max_iter or Config.SPCA_MAX_ITER
        self.n_neighbors = n_neighbors or Config.KNN_NEIGHBORS
        self.metric = metric or Config.KNN_METRIC
        
        # 构建处理流水线
        self._build_pipeline()
        
        # 协方差矩阵缓存
        self.cov_matrix = None

    def _build_pipeline(self):
        """构建稀疏PCA+KNN处理流水线"""
        self.pipeline = Pipeline([
            ('spca', SparsePCA(
                n_components=self.n_components,
                alpha=self.alpha,
                max_iter=self.max_iter,
                random_state=Config.RANDOM_STATE
            )),
            ('knn', KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                metric=self.metric,
                metric_params={'VI': None}  # 延迟到fit阶段设置
            ))
        ])

    def fit(self, X, y):
        """训练模型"""
        # 训练稀疏PCA
        X_spca = self.pipeline.named_steps['spca'].fit_transform(X)
        
        # 计算协方差矩阵（正则化处理）
        self.cov_matrix = np.cov(X_spca, rowvar=False)
        np.fill_diagonal(self.cov_matrix, self.cov_matrix.diagonal() + 1e-6)  # 正则化
        
        # 更新KNN参数
        self.pipeline.named_steps['knn'].metric_params = {
            'VI': np.linalg.inv(self.cov_matrix)
        }
        
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        """执行预测"""
        return self.pipeline.predict(X)
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        y_pred = self.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred, zero_division=0)
        }
    
    def save_model(self, path):
        """保存模型到文件"""
        joblib.dump({
            'pipeline': self.pipeline,
            'cov_matrix': self.cov_matrix,
            'params': {
                'n_components': self.n_components,
                'alpha': self.alpha,
                'max_iter': self.max_iter,
                'n_neighbors': self.n_neighbors,
                'metric': self.metric
            }
        }, path)
    
    @classmethod
    def load_model(cls, path):
        """从文件加载模型"""
        data = joblib.load(path)
        model = cls(
            n_components=data['params']['n_components'],
            alpha=data['params']['alpha'],
            max_iter=data['params']['max_iter'],
            n_neighbors=data['params']['n_neighbors'],
            metric=data['params']['metric']
        )
        model.pipeline = data['pipeline']
        model.cov_matrix = data['cov_matrix']
        return model

    @property
    def components_(self):
        """获取稀疏PCA成分"""
        return self.pipeline.named_steps['spca'].components_