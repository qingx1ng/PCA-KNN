import dataclasses

@dataclasses.dataclass
class Config:
    # 数据路径
    DATA_PATH: str = "data"
    PROCESSED_DATA_PATH: str = "data/processed"
    
    # 模型参数
    SPCA_COMPONENTS: int = 50       # 稀疏PCA主成分数
    SPCA_ALPHA: float = 0.1         # L1正则化系数（控制稀疏性）
    SPCA_MAX_ITER: int = 1000       # 最大迭代次数
    KNN_NEIGHBORS: int = 5          # KNN的K值
    KNN_METRIC: str = "mahalanobis" # 距离度量方式
    
    # 训练测试分割
    TEST_SIZE: float = 0.2          # 测试集比例
    RANDOM_STATE: int = 42          # 随机种子
    
    # 可视化增强参数
    PLOT_SAVE_PATH: str = "results/figures"
    N_EIGENFACES_TO_SHOW: int = 16  # 显示的特征脸数量
    SPARSE_THRESHOLD: float = 0.01  # 稀疏成分可视化阈值
    VARIANCE_THRESHOLD: float = 0.95 # 方差解释率阈值