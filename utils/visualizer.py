import matplotlib.pyplot as plt
import numpy as np
import os
from config.settings import Config
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def visualize_eigenfaces(pca, image_shape=(92, 112)):
    """
    可视化特征脸
    """
    plt.figure(figsize=(10, 8))
    
    # 显示平均脸
    plt.subplot(4, 4, 1)
    plt.imshow(pca.mean_.reshape(image_shape), cmap='gray')
    plt.title("Mean Face")
    plt.axis('off')
    
    # 显示前n_components-1个特征脸
    for i in range(1, Config.N_EIGENFACES_TO_SHOW):
        plt.subplot(4, 4, i+1)
        eigenface = pca.components_[i-1].reshape(image_shape)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f"Eigenface {i}")
        plt.axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(Config.PLOT_SAVE_PATH, exist_ok=True)
    plt.savefig(os.path.join(Config.PLOT_SAVE_PATH, 'eigenfaces.png'))
    plt.close()

def plot_reconstruction(pca, sample, image_shape=(112, 92)):
    """
    显示不同主成分数量下的重建效果
    """
    print('1--------------------------------------------')
    plt.figure(figsize=(12, 6))
    n_components_to_try = [10, 30, 50, 100, 150, 200]
    
    for i, n in enumerate(n_components_to_try):
        # 使用前n个主成分重建
        pca_temp = PCA(n_components=n).fit(sample)
        transformed = pca_temp.transform([sample])
        reconstructed = pca_temp.inverse_transform(transformed)
        
        plt.subplot(2, 3, i+1)
        plt.imshow(reconstructed.reshape(image_shape), cmap='gray')
        plt.title(f"{n} components")
        plt.axis('off')
    
    plt.suptitle("Reconstruction with Different PCA Components")
    plt.tight_layout()
    
    # 保存图像
    print('1--------------------------------------------')
    plt.savefig(os.path.join(Config.PLOT_SAVE_PATH, 'reconstruction.png'))
    plt.close()


def plot_pca_variance(X, threshold=0.95):
    """
    绘制PCA累计方差解释率曲线，并标记阈值线。

    参数:
        X (array): 输入数据（未降维）.
        threshold (float): 标记的目标方差阈值（默认0.95）.
    """
    pca = PCA().fit(X)
    plt.figure(figsize=(8, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), 'b-')
    plt.axhline(y=threshold, color='r', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'PCA Explained Variance (Threshold={threshold})')
    plt.grid()
    plt.tight_layout()

    # 保存图像
    os.makedirs(Config.PLOT_SAVE_PATH, exist_ok=True)
    plt.savefig(os.path.join(Config.PLOT_SAVE_PATH, 'pca_variance.png'))
    plt.close()


def plot_pca_scatter(X, y, n_components=2, alpha=0.6):
    """
    用前2个或3个主成分绘制数据分布散点图。

    参数:
        X (array): 输入数据.
        y (array): 标签（用于着色）.
        n_components (int): 2或3，表示降维后的维度.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    if n_components == 2:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=alpha, cmap='viridis')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
    elif n_components == 3:
        ax = plt.axes(projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, alpha=alpha, cmap='viridis')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
    plt.colorbar()
    plt.title(f'{n_components}D PCA Projection')
    plt.tight_layout()

    # 保存图像
    os.makedirs(Config.PLOT_SAVE_PATH, exist_ok=True)
    plt.savefig(os.path.join(Config.PLOT_SAVE_PATH, 'pca_scatter.png'))
    plt.close()


def plot_knn_k_selection(X, y, k_range=range(1, 20), cv=5):
    """
    分析不同K值对交叉验证准确率的影响。

    参数:
        X (array): 输入数据（建议已降维）.
        y (array): 标签.
        k_range (range): K值范围（默认1到20）.
        cv (int): 交叉验证折数.
    """
    accuracies = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
        accuracies.append(scores.mean())

    plt.figure(figsize=(8, 4))
    plt.plot(k_range, accuracies, 'bo-')
    plt.xlabel('K Value')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('KNN Accuracy vs K')
    plt.grid()
    plt.tight_layout()

    # 保存图像
    os.makedirs(Config.PLOT_SAVE_PATH, exist_ok=True)
    plt.savefig(os.path.join(Config.PLOT_SAVE_PATH, 'knn_k_selection.png'))
    plt.close()
