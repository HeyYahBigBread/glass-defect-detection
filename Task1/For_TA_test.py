import numpy as np
import os
import glob
import cv2
from PIL import Image
import torch
import argparse
from sklearn.metrics import f1_score

# 学生ID（需要替换）
STUDENT_ID = "PB23000243"


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


def preprocess_image(img_path):
    """预处理单个图像"""
    img = Image.open(img_path)
    img = img.resize((85, 85))  # 降采样
    img = np.array(img).astype(np.float32) / 255.0

    # 转换为灰度图
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 提取HOG特征
    features = extract_hog_features(img)
    return features


class NeuralNetwork:
    """简化版神经网络，只包含推理所需的方法"""

    def __init__(self):
        self.weights = []
        self.biases = []
        self.layer_sizes = []

    def relu(self, z):
        return np.maximum(0, z)

    def sigmoid(self, z):
        z = np.clip(z, -100, 100)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        a = X.T

        # 隐藏层
        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], a) + self.biases[i]
            a = self.relu(z)

        # 输出层
        z = np.dot(self.weights[-1], a) + self.biases[-1]
        a = self.sigmoid(z)

        return a.T

    def predict(self, X):
        return self.forward(X).flatten()

    def predict_classes(self, X, threshold=0.5):
        probs = self.predict(X)
        return (probs >= threshold).astype(int)

    def load_model(self, filename):
        """加载模型参数"""
        model_params = torch.load(filename, map_location='cpu')
        self.weights = model_params['weights']
        self.biases = model_params['biases']
        self.layer_sizes = model_params['layer_sizes']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True, help='测试数据路径')
    args = parser.parse_args()

    # 加载训练好的模型
    model = NeuralNetwork()
    model.load_model("glass_defect_model.pkl")

    # 获取测试数据
    img_dir = os.path.join(args.test_data_path, "img")
    txt_dir = os.path.join(args.test_data_path, "txt")
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    # 准备数据和真实标签
    X_test = []
    y_true = []

    # 记录开始时间
    start_time = time.time()

    # 处理每个图像
    for img_file in img_files:
        # 预处理图像
        features = preprocess_image(img_file)
        X_test.append(features)

        # 确定真实标签
        base_name = os.path.basename(img_file).replace(".png", ".txt")
        txt_file = os.path.join(txt_dir, base_name)
        label = 1 if os.path.exists(txt_file) else 0
        y_true.append(label)

    X_test = np.array(X_test)
    y_true = np.array(y_true)

    # 预测
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred >= 0.5).astype(int)

    # 计算F1分数
    f1 = f1_score(y_true, y_pred_classes)

    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"{STUDENT_ID}:{f1:.4f}")
    print(f"评估完成，共处理 {len(img_files)} 个图像，耗时 {elapsed_time:.2f} 秒", file=sys.stderr)


if __name__ == "__main__":
    import sys

    main()