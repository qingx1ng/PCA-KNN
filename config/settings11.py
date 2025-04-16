# 全局配置参数
class Config:
    # 数据路径
    DATA_PATH = "data"
    # PROCESSED_DATA_PATH = "data/processed"
    
    # 模型参数
    PCA_COMPONENTS = 50      # PCA主成分数量
    KNN_NEIGHBORS = 3       # KNN的K值
    
    # 训练测试分割
    TEST_SIZE = 0.2          # 测试集比例
    RANDOM_STATE = 40        # 随机种子
    
    # 可视化设置
    PLOT_SAVE_PATH = "results/figures"
    N_EIGENFACES_TO_SHOW = 16  # 显示的特征脸数量