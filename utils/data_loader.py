import os
import cv2
import numpy as np
from config.settings import Config

def load_orl_dataset():
    """
    加载ORL人脸数据集
    """
    X = []
    y = []
    
    # 统一使用正斜杠并确保路径正确
    data_path = Config.DATA_PATH.replace('\\', '/')
    print(f"正在从以下路径加载数据: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集路径不存在: {data_path}")

    for subject in sorted(os.listdir(data_path)):
        subject_dir = os.path.join(data_path, subject).replace('\\', '/')
        if not os.path.isdir(subject_dir):
            continue
            
        print(f"处理文件夹: {subject_dir}")
        
        for image_file in sorted(os.listdir(subject_dir)):
            # 更灵活的文件扩展名处理
            if not image_file.lower().endswith(('.pgm', '.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(subject_dir, image_file).replace('\\', '/')
            try:
                # 使用cv2.imread时确保路径是字符串(不是unicode)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"警告: 无法读取图像(可能路径/权限问题): {image_path}")
                    continue
                
                img = cv2.resize(img, (92, 112))
                X.append(img.flatten())
                y.append(int(subject[1:]) - 1)  # s1 -> 0, s2 -> 1等
                
            except Exception as e:
                print(f"处理图像时出错 {image_path}: {str(e)}")
                continue
    
    if len(X) == 0:
        raise ValueError("没有加载到任何数据，请检查数据集路径和文件格式")
    
    print(f"成功加载 {len(X)} 个样本")
    return np.array(X), np.array(y)

def preprocess_data(X, y):
    """
    数据预处理
    包括归一化、训练测试分割等
    """
    # 归一化到0-1
    X = X / 255.0
    
    # 划分训练测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE, 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test