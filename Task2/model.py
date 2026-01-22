import torch
import torch.nn as nn
import torchvision.models as models


class MultiLabelResNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return torch.sigmoid(self.resnet(x))