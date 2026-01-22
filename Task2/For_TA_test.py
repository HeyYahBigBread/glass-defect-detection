import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import GlassDefectDataset
from model import MultiLabelResNet
from evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True)
    args = parser.parse_args()

    # 加载模型
    model = MultiLabelResNet()
    model.load_state_dict(torch.load('saved_model/multi_label_resnet.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = GlassDefectDataset(
        img_dir=f'{args.test_data_path}/img',
        txt_dir=f'{args.test_data_path}/txt',
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 评估
    micro_f1 = evaluate_model(model, test_loader, device)

    # 输出格式：学号:分数
    print("P823000243:" + f"{micro_f1:.4f}")


if __name__ == '__main__':
    main()