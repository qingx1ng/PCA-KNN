import matplotlib.pyplot as plt
import numpy as np
import os
from config.settings import Config
from sklearn.decomposition import PCA

def visualize_eigenfaces(pca, image_shape=(112, 92)):
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
    plt.savefig(os.path.join(Config.PLOT_SAVE_PATH, 'reconstruction.png'))
    plt.close()