import os
import glob
import numpy as np
import cv2
from PIL import Image
import random
from sklearn.model_selection import train_test_split


def load_data(data_path):
    """加载图像数据和对应标签"""
    img_dir = os.path.join(data_path, "img")
    txt_dir = os.path.join(data_path, "txt")

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    X = []
    y = []

    for img_file in img_files:
        # 加载并预处理图像
        img = Image.open(img_file)
        img = img.resize((85, 85))  # 降采样减少计算量
        img = np.array(img).astype(np.float32) / 255.0

        # 转换为灰度图
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 提取HOG特征
        features = extract_hog_features(img)

        # 确定标签
        base_name = os.path.basename(img_file).replace(".png", ".txt")
        txt_file = os.path.join(txt_dir, base_name)
        label = 1 if os.path.exists(txt_file) else 0  # 1=有缺陷, 0=无缺陷

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


def extract_hog_features(image, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    """手动实现简化版HOG特征提取"""
    # 1. 计算图像梯度
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    # 2. 计算梯度幅度和方向
    mag, ang = cv2.cartToPolar(gx, gy)

    # 3. 量化方向到bins
    bins = np.int32(nbins * ang / (2 * np.pi))

    # 4. 创建HOG单元格
    n_cells = (image.shape[0] // cell_size[0], image.shape[1] // cell_size[1])
    histograms = np.zeros((n_cells[0], n_cells[1], nbins))

    # 5. 为每个单元格计算方向直方图
    for i in range(n_cells[0]):
        for j in range(n_cells[1]):
            cell_mag = mag[i * cell_size[0]:(i + 1) * cell_size[0], j * cell_size[1]:(j + 1) * cell_size[1]]
            cell_bins = bins[i * cell_size[0]:(i + 1) * cell_size[0], j * cell_size[1]:(j + 1) * cell_size[1]]

            for k in range(nbins):
                histograms[i, j, k] = np.sum(cell_mag[cell_bins == k])

    # 6. 归一化块
    hog_features = []
    for i in range(n_cells[0] - block_size[0] + 1):
        for j in range(n_cells[1] - block_size[1] + 1):
            block = histograms[i:i + block_size[0], j:j + block_size[1]]
            block_norm = np.linalg.norm(block.ravel() + 1e-6)
            hog_features.extend((block.ravel() / block_norm).tolist())

    return np.array(hog_features)


def handle_class_imbalance(X, y):
    """过采样少数类（有缺陷的图像）"""
    # 找出有缺陷和无缺陷的样本
    defective_indices = np.where(y == 1)[0]
    non_defective_indices = np.where(y == 0)[0]

    # 计算需要复制的样本数量
    n_defective = len(defective_indices)
    n_non_defective = len(non_defective_indices)

    if n_defective < n_non_defective:
        # 过采样有缺陷的样本
        additional_indices = np.random.choice(defective_indices, n_non_defective - n_defective, replace=True)
        X_additional = X[additional_indices]
        y_additional = y[additional_indices]

        # 合并数据
        X_balanced = np.vstack([X, X_additional])
        y_balanced = np.hstack([y, y_additional])

        # 打乱数据
        indices = np.random.permutation(len(X_balanced))
        return X_balanced[indices], y_balanced[indices]

    return X, y


def train_val_split(X, y, val_ratio=0.2):
    """划分训练集和验证集"""
    return train_test_split(X, y, test_size=val_ratio, random_state=42, stratify=y)