import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import cv2
from sklearn.metrics import f1_score
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')
import time
import copy


class GlassDefectDataset(Dataset):
    """玻璃缺陷数据集"""

    def __init__(self, data_path, transform=None):
        self.img_dir = os.path.join(data_path, "img")
        self.txt_dir = os.path.join(data_path, "txt")
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img = Image.open(img_file).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # 获取标签（4个类别：无缺陷、边缘缺口、划痕、污渍）
        labels = np.zeros(4)

        base_name = os.path.basename(img_file).replace(".png", ".txt")
        txt_file = os.path.join(self.txt_dir, base_name)

        # 如果没有标签文件，则为无缺陷
        if not os.path.exists(txt_file):
            labels[0] = 1  # 无缺陷
        else:
            # 读取标签文件
            with open(txt_file, 'r') as f:
                lines = f.readlines()

            # 标记存在的缺陷类型
            defect_types = set()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    if 0 <= class_id <= 2:  # 0:边缘缺口, 1:划痕, 2:污渍
                        defect_types.add(class_id + 1)  # +1因为索引从1开始（0是无缺陷）

            # 设置标签
            if len(defect_types) > 0:
                # 有缺陷，标记对应的缺陷类型
                for defect_type in defect_types:
                    if defect_type < 4:  # 确保索引在范围内
                        labels[defect_type] = 1
            else:
                # 没有检测到缺陷类型，标记为无缺陷
                labels[0] = 1

        return img, torch.tensor(labels, dtype=torch.float32)


class MultiLabelClassifier(nn.Module):
    """多标签分类模型"""

    def __init__(self, num_classes=4):
        super(MultiLabelClassifier, self).__init__()

        # 使用预训练的ResNet18作为骨干网络
        self.backbone = models.resnet18(pretrained=True)

        # 修改最后一层
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25):
    """训练模型"""
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 每个epoch有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()  # 设置模型为评估模式

            running_loss = 0.0
            all_preds = []
            all_labels = []

            # 遍历数据
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 零梯度
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs)  # 应用sigmoid获取概率
                    loss = criterion(outputs, labels)

                    # 预测
                    preds = (probs > 0.5).float()

                    # 反向传播 + 优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            # 计算Micro F1分数
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            epoch_f1 = f1_score(all_labels, all_preds, average='micro')

            print(f'{phase} Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f}')

            # 记录历史
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_f1'].append(epoch_f1)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_f1'].append(epoch_f1)

            # 深度复制模型
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_model_task2.pth')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val F1: {best_f1:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../dataset/train', help='训练数据路径')
    args = parser.parse_args()

    # 数据增强和预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 创建数据集
    full_dataset = GlassDefectDataset(args.data_path, transform=data_transforms['train'])

    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 为验证集设置不同的transform
    val_dataset.dataset.transform = data_transforms['val']

    # 创建数据加载器
    batch_size = 32
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化模型
    model = MultiLabelClassifier(num_classes=4)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 训练模型
    model, history, best_f1 = train_model(
        model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25
    )

    # 保存最终模型
    torch.save(model.state_dict(), 'final_model_task2.pth')

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
    plt.plot(history['train_f1'], label='训练F1')
    plt.plot(history['val_f1'], label='验证F1')
    plt.title('F1分数曲线')
    plt.xlabel('Epoch')
    plt.ylabel('F1分数')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history_task2.png')
    print("训练历史已保存到 training_history_task2.png")


if __name__ == "__main__":
    main()