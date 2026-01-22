import os
import argparse
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import sys

# --- 配置 (必须与 main.py 训练时完全一致) ---
IMG_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STUDENT_ID = "PB23000000"  # TODO: 请务必修改为你的真实学号

# ==============================================================================
# 1. 模型定义 (必须复制自 main.py，确保结构完全一致)
# ==============================================================================

class ManualConv2d:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 初始化占位 (会被 load 覆盖)
        self.W = torch.zeros(out_channels, in_channels * kernel_size * kernel_size).to(DEVICE)
        self.b = torch.zeros(1, out_channels).to(DEVICE)

    def forward(self, x):
        N, C, H, W = x.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # im2col
        inp_unf = F.unfold(x, (self.kernel_size, self.kernel_size), padding=self.padding, stride=self.stride)
        
        # Matrix Multiplication: (N, L, Cin*K*K) @ W.t()
        out = inp_unf.transpose(1, 2).matmul(self.W.t()) 
        out += self.b 
        
        # Reshape
        out = out.transpose(1, 2).view(N, self.out_channels, H_out, W_out)
        return out

class ManualMaxPool2d:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, x):
        out, _ = F.max_pool2d(x, self.kernel_size, self.stride, return_indices=True)
        return out

class ManualFlatten:
    def forward(self, x):
        return x.view(x.shape[0], -1)

class ManualReLU:
    def forward(self, x):
        return x * (x > 0).float()

class ManualLinear:
    def __init__(self, in_features, out_features):
        self.W = torch.zeros(in_features, out_features).to(DEVICE)
        self.b = torch.zeros(1, out_features).to(DEVICE)
    def forward(self, x):
        return x.mm(self.W) + self.b

class ManualSigmoid:
    def forward(self, x):
        return 1.0 / (1.0 + torch.exp(-torch.clamp(x, -50, 50)))

class CNN:
    def __init__(self):
        # 结构定义必须与训练时完全一致
        self.layers = [
            ManualConv2d(1, 8, kernel_size=5, stride=1, padding=2),
            ManualReLU(),
            ManualMaxPool2d(2, 2),
            
            ManualConv2d(8, 16, kernel_size=3, stride=1, padding=1),
            ManualReLU(),
            ManualMaxPool2d(2, 2),
            
            ManualConv2d(16, 32, kernel_size=3, stride=1, padding=1),
            ManualReLU(),
            ManualMaxPool2d(2, 2),
            
            ManualFlatten(),
            ManualLinear(32 * 16 * 16, 128),
            ManualReLU(),
            ManualLinear(128, 1),
            ManualSigmoid()
        ]

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def load(self, path):
        # 加载权重
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        checkpoint = torch.load(path, map_location=DEVICE)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (ManualConv2d, ManualLinear)):
                layer.W = checkpoint[f'{i}_W'].to(DEVICE)
                layer.b = checkpoint[f'{i}_b'].to(DEVICE)

# ==============================================================================
# 2. 评测逻辑
# ==============================================================================

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True)
    args = parser.parse_args()

    img_dir = os.path.join(args.test_data_path, 'img')
    txt_dir = os.path.join(args.test_data_path, 'txt')
    
    img_paths = glob.glob(os.path.join(img_dir, '*.png'))
    if len(img_paths) == 0:
        print(f"{STUDENT_ID}:0.0")
        return

    # 1. 加载模型
    model = CNN()
    # 假设模型文件在当前脚本同级目录下
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_cnn.pth')
    
    try:
        model.load(model_path)
    except Exception as e:
        # 为了符合格式要求，这里可以打印错误到 stderr，但 stdout 必须保持格式
        sys.stderr.write(f"Error loading model: {e}\n")
        print(f"{STUDENT_ID}:0.0")
        return

    preds_list = []
    labels_list = []
    
    # 2. 批量推理 (Batch Inference) 以加快速度
    BATCH_SIZE = 32
    batch_imgs = []
    
    # 获取标签并进行推理
    for i, img_path in enumerate(img_paths):
        # --- 预处理 ---
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        # CNN 输入格式: (C, H, W)
        img_tensor = img[np.newaxis, :, :] 
        batch_imgs.append(img_tensor)
        
        # --- 读取真实标签 (用于计算 F1) ---
        base_name = os.path.basename(img_path).replace('.png', '')
        txt_path = os.path.join(txt_dir, base_name + '.txt')
        label = 1 if os.path.exists(txt_path) else 0
        labels_list.append(label)
        
        # --- 执行推理 ---
        if len(batch_imgs) == BATCH_SIZE or i == len(img_paths) - 1:
            # 转换为 Tensor: (Batch, C, H, W)
            batch_np = np.array(batch_imgs)
            batch_t = torch.tensor(batch_np).float().to(DEVICE)
            
            with torch.no_grad():
                outputs = model.forward(batch_t)
                # 使用 0.5 作为阈值 (或者你可以硬编码训练时得到的最佳阈值，例如 0.4)
                # outputs shape: (Batch, 1) -> flatten
                batch_preds = (outputs > 0.5).cpu().numpy().flatten().astype(int)
                preds_list.extend(batch_preds)
            
            batch_imgs = []

    # 3. 计算 F1 分数
    # pos_label=1 指定计算缺陷类别的 F1
    score = f1_score(labels_list, preds_list, pos_label=1)
    
    # 4. 严格按照格式输出: 学号:F1
    print(f"{STUDENT_ID}:{score:.4f}")

if __name__ == '__main__':
    evaluate()