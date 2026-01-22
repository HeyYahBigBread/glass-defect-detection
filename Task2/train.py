import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import time


def train_model(model, train_loader, device, epochs=20, lr=0.001, weight_decay=1e-4):
    """
    训练多标签分类模型
    """
    # 损失函数：二元交叉熵（适用于多标签分类）
    criterion = nn.BCELoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 学习率调度器
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # 记录训练过程
    train_losses = []
    train_f1_scores = []

    print(f"开始训练，设备: {device}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {train_loader.batch_size}")
    print("-" * 50)

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        # 训练循环
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for batch_idx, (images, labels) in enumerate(pbar):
                # 移动数据到设备
                images = images.to(device)
                labels = labels.to(device)

                # 前向传播
                optimizer.zero_grad()
                outputs = model(images)

                # 计算损失
                loss = criterion(outputs, labels)

                # 反向传播
                loss.backward()
                optimizer.step()

                # 记录损失
                running_loss += loss.item()

                # 收集预测和标签用于计算F1分数
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().detach().numpy())
                all_labels.extend(labels.cpu().detach().numpy())

                # 更新进度条
                pbar.set_postfix({'loss': loss.item()})

        # 计算平均损失和F1分数
        avg_loss = running_loss / len(train_loader)

        # 计算Micro F1分数
        if len(all_preds) > 0:
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            micro_f1 = f1_score(all_labels, all_preds, average='micro')
        else:
            micro_f1 = 0.0

        # 记录结果
        train_losses.append(avg_loss)
        train_f1_scores.append(micro_f1)

        # 更新学习率
        scheduler.step()

        # 打印epoch结果
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Loss: {avg_loss:.4f} - "
              f"Micro F1: {micro_f1:.4f} - "
              f"Time: {epoch_time:.2f}s - "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # 每5个epoch保存一次模型
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'saved_model/multi_label_resnet_epoch{epoch + 1}.pth')
            print(f"模型已保存到 saved_model/multi_label_resnet_epoch{epoch + 1}.pth")

    # 保存最终模型
    torch.save(model.state_dict(), 'saved_model/multi_label_resnet_final.pth')
    print("最终模型已保存到 saved_model/multi_label_resnet_final.pth")

    return train_losses, train_f1_scores


def validate_model(model, val_loader, device):
    """
    验证模型性能
    """
    model.eval()
    criterion = nn.BCELoss()

    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='验证中', unit='batch'):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 收集预测和标签
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = val_loss / len(val_loader)

    # 计算各个指标
    if len(all_preds) > 0:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 计算每个类别的F1分数
        class_f1 = f1_score(all_labels, all_preds, average=None)

        # 计算Micro和Macro F1
        micro_f1 = f1_score(all_labels, all_preds, average='micro')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        # 计算准确率
        accuracy = (all_preds == all_labels).mean()

        return {
            'loss': avg_loss,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'accuracy': accuracy,
            'class_f1': class_f1
        }

    return {'loss': avg_loss, 'micro_f1': 0.0, 'macro_f1': 0.0, 'accuracy': 0.0, 'class_f1': []}