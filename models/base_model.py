from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from config.settings import Config

class PCA_KNN_Model:
    def __init__(self, n_components=None, n_neighbors=None):
        """
        初始化PCA+KNN模型
        """
        self.n_components = n_components or Config.PCA_COMPONENTS
        self.n_neighbors = n_neighbors or Config.KNN_NEIGHBORS
        
        self.pca = PCA(n_components=self.n_components, svd_solver='auto')
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
    
    def fit(self, X, y):
        """训练模型"""
        # PCA拟合和转换
        X_pca = self.pca.fit_transform(X)
        # KNN训练
        self.knn.fit(X_pca, y)
        return self
    
    def predict(self, X):
        """预测"""
        X_pca = self.pca.transform(X)
        return self.knn.predict(X_pca)
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        metrics = {
            'accuracy': acc,
            'report': report
        }
        return metrics
    
    def save_model(self, path):
        """保存模型"""
        import joblib
        joblib.dump({
            'pca': self.pca,
            'knn': self.knn,
            'params': {
                'n_components': self.n_components,
                'n_neighbors': self.n_neighbors
            }
        }, path)
    
    @classmethod
    def load_model(cls, path):
        """加载模型"""
        import joblib
        data = joblib.load(path)
        model = cls(
            n_components=data['params']['n_components'],
            n_neighbors=data['params']['n_neighbors']
        )
        model.pca = data['pca']
        model.knn = data['knn']
        return model