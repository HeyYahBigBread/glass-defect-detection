import numpy as np
import math
import random
import torch


class NeuralNetwork:
    """从零实现的简单全连接神经网络"""

    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, momentum=0.9):
        """
        初始化网络参数
        input_size: 输入特征维度
        hidden_sizes: 隐藏层大小列表
        output_size: 输出维度
        learning_rate: 学习率
        momentum: 动量参数
        """
        self.learning_rate = learning_rate
        self.momentum = momentum

        # 初始化网络结构
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(self.layer_sizes) - 1

        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        self.velocity_w = []  # 用于动量
        self.velocity_b = []  # 用于动量

        for i in range(self.num_layers):
            input_dim = self.layer_sizes[i]
            output_dim = self.layer_sizes[i + 1]

            # He初始化
            scale = math.sqrt(2.0 / input_dim)
            w = np.random.randn(output_dim, input_dim) * scale
            b = np.zeros((output_dim, 1))

            self.weights.append(w)
            self.biases.append(b)
            self.velocity_w.append(np.zeros_like(w))
            self.velocity_b.append(np.zeros_like(b))

        # 保存中间计算结果用于反向传播
        self.activations = []
        self.z_values = []

    def relu(self, z):
        """ReLU激活函数"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """ReLU导数"""
        return (z > 0).astype(float)

    def sigmoid(self, z):
        """Sigmoid激活函数"""
        # 防止溢出
        z = np.clip(z, -100, 100)
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        """Sigmoid导数（用激活值计算更稳定）"""
        return a * (1 - a)

    def forward(self, X):
        """前向传播"""
        # 确保X是二维数组
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 转置X以便与权重矩阵相乘
        a = X.T
        self.activations = [a]  # 保存输入
        self.z_values = []

        # 隐藏层（使用ReLU）
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], a) + self.biases[i]
            self.z_values.append(z)
            a = self.relu(z)
            self.activations.append(a)

        # 输出层（使用Sigmoid）
        z = np.dot(self.weights[-1], a) + self.biases[-1]
        self.z_values.append(z)
        a = self.sigmoid(z)
        self.activations.append(a)

        # 转置回样本在第一维
        return a.T

    def backward(self, X, y_true, y_pred):
        """反向传播，手动计算梯度"""
        m = X.shape[0]

        # 计算输出层误差
        delta = y_pred - y_true.reshape(-1, 1)
        delta = delta.T  # 转置以匹配内部表示

        # 存储权重和偏置的梯度
        grad_w = [None] * self.num_layers
        grad_b = [None] * self.num_layers

        # 输出层梯度
        a_prev = self.activations[-2]
        grad_w[-1] = np.dot(delta, a_prev.T) / m
        grad_b[-1] = np.sum(delta, axis=1, keepdims=True) / m

        # 反向传播到隐藏层
        for l in range(self.num_layers - 2, -1, -1):
            z = self.z_values[l]
            a_prev = self.activations[l]

            # 计算当前层的误差
            delta = np.dot(self.weights[l + 1].T, delta) * self.relu_derivative(z)

            # 计算梯度
            grad_w[l] = np.dot(delta, a_prev.T) / m
            grad_b[l] = np.sum(delta, axis=1, keepdims=True) / m

        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b):
        """使用带动量的SGD更新参数"""
        for i in range(self.num_layers):
            # 更新动量
            self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.learning_rate * grad_w[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.learning_rate * grad_b[i]

            # 更新参数
            self.weights[i] += self.velocity_w[i]
            self.biases[i] += self.velocity_b[i]

    def compute_loss(self, y_true, y_pred):
        """计算二元交叉熵损失"""
        # 防止数值不稳定
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        y_true = y_true.reshape(-1, 1)

        # 二元交叉熵
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, patience=5):
        """训练模型，带早停机制"""
        best_val_f1 = 0
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size

        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0

            # 小批量训练
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # 前向传播
                y_pred = self.forward(X_batch)

                # 计算损失
                loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += loss

                # 反向传播
                grad_w, grad_b = self.backward(X_batch, y_batch, y_pred)

                # 更新参数
                self.update_parameters(grad_w, grad_b)

            # 计算平均训练损失
            epoch_loss /= n_batches
            history['train_loss'].append(epoch_loss)

            # 在验证集上评估
            val_pred = self.predict(X_val)
            val_loss = self.compute_loss(y_val, val_pred)
            val_f1 = self.calculate_f1(y_val, val_pred)

            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

            # 早停机制
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                # 保存最佳模型
                self.save_model("best_model.pkl")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # 加载最佳模型
        self.load_model("best_model.pkl")
        return history

    def predict(self, X):
        """预测概率"""
        return self.forward(X).flatten()

    def predict_classes(self, X, threshold=0.5):
        """预测类别"""
        probs = self.predict(X)
        return (probs >= threshold).astype(int)

    def calculate_f1(self, y_true, y_pred, threshold=0.5):
        """计算F1分数"""
        y_pred_classes = (y_pred >= threshold).astype(int)

        # 真阳性、假阳性、假阴性
        tp = np.sum((y_true == 1) & (y_pred_classes == 1))
        fp = np.sum((y_true == 0) & (y_pred_classes == 1))
        fn = np.sum((y_true == 1) & (y_pred_classes == 0))

        # 精确率和召回率
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)

        # F1分数
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        return f1

    def save_model(self, filename):
        """保存模型参数"""
        model_params = {
            'weights': self.weights,
            'biases': self.biases,
            'layer_sizes': self.layer_sizes
        }
        torch.save(model_params, filename)

    def load_model(self, filename):
        """加载模型参数"""
        model_params = torch.load(filename, weights_only=False)
        self.weights = model_params['weights']
        self.biases = model_params['biases']
        self.layer_sizes = model_params['layer_sizes']
        self.num_layers = len(self.layer_sizes) - 1