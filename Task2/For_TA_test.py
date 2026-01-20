import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
from tqdm import tqdm
import sys
import time
import cv2

# 学生ID（需要替换）
STUDENT_ID = "PB23000243"


class MultiLabelClassifier(nn.Module):
    """多标签分类模型"""

    def __init__(self, num_classes=4):
        super(MultiLabelClassifier, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class GlassDefectDatasetTest:
    """用于测试的玻璃缺陷数据集"""

    def __init__(self, data_path):
        self.img_dir = os.path.join(data_path, "img")
        self.txt_dir = os.path.join(data_path, "txt")
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))

        # 预处理转换
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img = Image.open(img_file).convert('RGB')
        img = self.transform(img)
        return img, img_file

    def get_true_labels(self):
        """获取真实标签"""
        true_labels = []

        for img_file in self.img_files:
            base_name = os.path.basename(img_file).replace(".png", ".txt")
            txt_file = os.path.join(self.txt_dir, base_name)

            labels = np.zeros(4)  # 4个类别

            if not os.path.exists(txt_file):
                labels[0] = 1  # 无缺陷
            else:
                with open(txt_file, 'r') as f:
                    lines = f.readlines()

                defect_types = set()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])
                        if 0 <= class_id <= 2:
                            defect_types.add(class_id + 1)  # +1因为索引从1开始

                if len(defect_types) > 0:
                    for defect_type in defect_types:
                        if defect_type < 4:
                            labels[defect_type] = 1
                else:
                    labels[0] = 1

            true_labels.append(labels)

        return np.array(true_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True, help='测试数据路径')
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = MultiLabelClassifier(num_classes=4)
    model.load_state_dict(torch.load('final_model_task2.pth', map_location=device))
    model = model.to(device)
    model.eval()

    # 创建数据集
    dataset = GlassDefectDatasetTest(args.test_data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 获取真实标签
    y_true = dataset.get_true_labels()

    # 预测
    all_preds = []
    start_time = time.time()

    with torch.no_grad():
        for inputs, _ in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).cpu().numpy()
            all_preds.extend(preds)

    # 转换为numpy数组
    y_pred = np.array(all_preds)

    # 计算Micro F1分数
    from sklearn.metrics import f1_score
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    # 计算耗时
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 按要求格式输出
    print(f"{STUDENT_ID}:{micro_f1:.4f}")

    # 将耗时和处理图像数量输出到stderr（不会影响评分）
    print(f"评估完成，共处理 {len(dataset)} 个图像，耗时 {elapsed_time:.2f} 秒", file=sys.stderr)


if __name__ == "__main__":
    main()