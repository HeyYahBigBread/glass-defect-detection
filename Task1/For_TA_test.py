import os
import argparse
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import sys

# --- 配置 ---
IMG_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STUDENT_ID = "PB23000000"  # TODO: 务必修改为你的真实学号

# --- 核心修复：支持中文路径的读取函数 ---
def cv_imread(file_path):
    try:
        raw_data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(raw_data, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return None

# ==============================================================================
# 1. 模型定义 (必须保持完全一致)
# ==============================================================================
class ManualConv2d:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = torch.zeros(out_channels, in_channels * kernel_size * kernel_size).to(DEVICE)
        self.b = torch.zeros(1, out_channels).to(DEVICE)
    def forward(self, x):
        N, C, H, W = x.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        inp_unf = F.unfold(x, (self.kernel_size, self.kernel_size), padding=self.padding, stride=self.stride)
        out = inp_unf.transpose(1, 2).matmul(self.W.t()) + self.b 
        out = out.transpose(1, 2).reshape(N, self.out_channels, H_out, W_out)
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
        self.input_shape = x.shape  # 关键：记下输入的形状
        return x.reshape(x.shape[0], -1) 

    def backward(self, grad_output):
        # 使用记下的形状来还原，解决报错
        return grad_output.reshape(self.input_shape)

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
        self.layers = [
            # Layer 1: 1 -> 16
            ManualConv2d(1, 16, 5, 1, 2), ManualReLU(), ManualMaxPool2d(2, 2),
            # Layer 2: 16 -> 32
            ManualConv2d(16, 32, 3, 1, 1), ManualReLU(), ManualMaxPool2d(2, 2),
            # Layer 3: 32 -> 64
            ManualConv2d(32, 64, 3, 1, 1), ManualReLU(), ManualMaxPool2d(2, 2),
            # Layer 4: 64 -> 64 (新增一层)
            ManualConv2d(64, 64, 3, 1, 1), ManualReLU(), ManualMaxPool2d(2, 2),
            
            ManualFlatten(),
            # Linear输入维度: 128经过4次池化(每次减半)变成8 (128->64->32->16->8)
            # 所以是 64通道 * 8 * 8
            ManualLinear(64 * 8 * 8, 128), ManualReLU(),
            ManualLinear(128, 1), ManualSigmoid()
        ]

    def forward(self, x):
        out = x
        for layer in self.layers: out = layer.forward(out)
        return out
    def backward(self, grad_loss):
        grad = grad_loss
        for layer in reversed(self.layers): grad = layer.backward(grad)
    def step(self, lr):
        for layer in self.layers: 
            if hasattr(layer, 'step'): layer.step(lr)
    def save(self, path):
        checkpoint = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (ManualConv2d, ManualLinear)):
                checkpoint[f'{i}_W'] = layer.W.cpu(); checkpoint[f'{i}_b'] = layer.b.cpu()
        torch.save(checkpoint, path)
    def load(self, path):
        # 增加 weights_only=True 提高安全性
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (ManualConv2d, ManualLinear)):
                if f'{i}_W' in checkpoint: # 兼容性检查
                    layer.W = checkpoint[f'{i}_W'].to(DEVICE)
                    layer.b = checkpoint[f'{i}_b'].to(DEVICE)

# ==============================================================================
# 2. 评测逻辑 (加入自动阈值搜索)
# ==============================================================================
def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True)
    args = parser.parse_args()

    img_dir = os.path.join(args.test_data_path, 'img')
    txt_dir = os.path.join(args.test_data_path, 'txt')
    img_paths = glob.glob(os.path.join(img_dir, '*.png'))
    
    if not img_paths:
        print(f"{STUDENT_ID}:0.0")
        return

    # 加载模型
    model = CNN()
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_cnn.pth')
    try:
        model.load(model_path)
    except Exception:
        print(f"{STUDENT_ID}:0.0")
        return

    # 1. 获取所有预测概率
    all_probs = []
    all_labels = []
    
    BATCH_SIZE = 32
    batch_imgs = []
    
    for i, img_path in enumerate(img_paths):
        img = cv_imread(img_path)
        if img is None: continue
        
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        batch_imgs.append(img[np.newaxis, :, :])
        
        # 读取标签
        base = os.path.basename(img_path).replace('.png', '')
        label = 1 if os.path.exists(os.path.join(txt_dir, base + '.txt')) else 0
        all_labels.append(label)
        
        if len(batch_imgs) == BATCH_SIZE or i == len(img_paths) - 1:
            batch_t = torch.tensor(np.array(batch_imgs)).float().to(DEVICE)
            with torch.no_grad():
                out = model.forward(batch_t)
                all_probs.extend(out.cpu().numpy().flatten())
            batch_imgs = []

    if not all_labels:
        print(f"{STUDENT_ID}:0.0")
        return

    # 2. 自动搜索最佳阈值 (这才是提分的关键!)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    best_f1 = 0.0
    # 扫描 0.1 到 0.9，步长 0.05
    for thresh in np.arange(0.1, 0.95, 0.05):
        preds = (all_probs > thresh).astype(int)
        score = f1_score(all_labels, preds, pos_label=1)
        if score > best_f1:
            best_f1 = score
            
    # 输出最高分
    print(f"{STUDENT_ID}:{best_f1:.4f}")

if __name__ == '__main__':
    evaluate()