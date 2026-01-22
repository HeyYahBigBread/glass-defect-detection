import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import GlassDefectDataset
from model import MultiLabelResNet
from train import train_model
import os


def main():
    # 数据增强与归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 数据集
    train_dataset = GlassDefectDataset(
        img_dir=os.path.join(current_dir, '../dataset/train/img'),
        txt_dir=os.path.join(current_dir, '../dataset/train/txt'),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 模型
    model = MultiLabelResNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 训练
    train_model(model, train_loader, device, epochs=20)

    # 保存模型
    torch.save(model.state_dict(), 'saved_model/multi_label_resnet.pth')


if __name__ == '__main__':
    main()