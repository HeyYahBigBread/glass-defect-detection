import os
import cv2
import torch
from torch.utils.data import Dataset


class GlassDefectDataset(Dataset):
    def __init__(self, img_dir, txt_dir, transform=None):
        self.img_dir = img_dir
        self.txt_dir = txt_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        txt_name = img_name.replace('.png', '.txt')
        txt_path = os.path.join(self.txt_dir, txt_name)

        # 初始化标签：[no_defect, chipped_edge, scratch, stain]
        labels = [1, 0, 0, 0]  # 默认无缺陷

        if os.path.exists(txt_path):
            labels = [0, 0, 0, 0]  # 有缺陷，清空无缺陷标记
            with open(txt_path, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    if class_id in [0, 1, 2]:
                        labels[class_id + 1] = 1  # 对应类别标记为1

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels, dtype=torch.float32)