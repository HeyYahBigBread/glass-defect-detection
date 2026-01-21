import numpy as np
import os
import argparse
from utils import load_data, handle_class_imbalance, train_val_split
from model import NeuralNetwork
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')
from sklearn.metrics import precision_score, recall_score, f1_score
import gc
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../dataset/train', help='训练数据路径')
    args = parser.parse_args()

    print("加载数据...")
    X, y = load_data(args.data_path)
    print(f"加载了 {len(X)} 个样本，特征维度: {X.shape[1]}")

    # 处理类不平衡
    print("处理类不平衡...")
    X_balanced, y_balanced = handle_class_imbalance(X, y)
    print(f"平衡后的样本数量: {len(X_balanced)}")

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_val_split(X_balanced, y_balanced)
    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}")

    # 初始化模型
    input_size = X_train.shape[1]
    model = NeuralNetwork(
        input_size=input_size,
        hidden_sizes=[128, 64],
        output_size=1,
        learning_rate=0.001,
        momentum=0.9
    )

    # 训练模型
    print("开始训练...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=64,
        patience=10
    )

    # 评估最终模型
    print("评估最终模型...")
    y_val_pred = model.predict(X_val)
    y_val_pred_classes = (y_val_pred >= 0.5).astype(int)

    precision = precision_score(y_val, y_val_pred_classes)
    recall = recall_score(y_val, y_val_pred_classes)
    f1 = f1_score(y_val, y_val_pred_classes)

    print(f"验证集结果 - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # 保存最终模型
    model.save_model("glass_defect_model.pkl")
    print("模型已保存到 glass_defect_model.pkl")

    # 绘制训练历史
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='验证F1')
    plt.title('F1分数曲线')
    plt.xlabel('Epoch')
    plt.ylabel('F1分数')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("训练历史已保存到 training_history.png")

    # 释放内存
    del X, y, X_balanced, y_balanced, X_train, y_train, X_val, y_val
    gc.collect()


if __name__ == "__main__":
    main()