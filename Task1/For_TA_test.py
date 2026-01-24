import os
import argparse
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import json
import sys

# --- 配置 ---
IMG_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STUDENT_ID = "PB23111669"  # TODO: 务必确认这是你的真实学号
THRESHOLD = 0.5            # 重要：因为没有标签无法搜索，这里固定为 0.5

# --- 支持中文路径读取 ---
def cv_imread(file_path):
    try:
        raw_data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(raw_data, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return None

# ==============================================================================
# 1. 模型定义 (必须与 main.py 的 4层结构 完全一致)
# ==============================================================================
class ManualConv2d:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # 仅占位，实际参数由 load 加载
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
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1) 
    # Inference 不需要 backward，但为了保持类定义一致性可以留着

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
        # 4层结构，与 main.py 保持一致
        self.layers = [
            # Layer 1: 1 -> 16
            ManualConv2d(1, 16, 5, 1, 2), ManualReLU(), ManualMaxPool2d(2, 2),
            # Layer 2: 16 -> 32
            ManualConv2d(16, 32, 3, 1, 1), ManualReLU(), ManualMaxPool2d(2, 2),
            # Layer 3: 32 -> 64
            ManualConv2d(32, 64, 3, 1, 1), ManualReLU(), ManualMaxPool2d(2, 2),
            # Layer 4: 64 -> 64
            ManualConv2d(64, 64, 3, 1, 1), ManualReLU(), ManualMaxPool2d(2, 2),
            
            ManualFlatten(),
            # Linear输入维度: 64 * 8 * 8
            ManualLinear(64 * 8 * 8, 128), ManualReLU(),
            ManualLinear(128, 1), ManualSigmoid()
        ]

    def forward(self, x):
        out = x
        for layer in self.layers: out = layer.forward(out)
        return out

    def load(self, path):
        if not os.path.exists(path): raise FileNotFoundError(f"Model file not found: {path}")
        # weights_only=True 保证安全性
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (ManualConv2d, ManualLinear)):
                if f'{i}_W' in checkpoint:
                    layer.W = checkpoint[f'{i}_W'].to(DEVICE)
                    layer.b = checkpoint[f'{i}_b'].to(DEVICE)

# ==============================================================================
# 2. 推理逻辑 (生成 JSON)
# ==============================================================================
def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True)
    args = parser.parse_args()

    # 路径处理
    img_dir = os.path.join(args.test_data_path, 'img')
    
    # 兼容性处理：防止助教直接给的是图片目录而不是根目录
    if not os.path.exists(img_dir):
        # 尝试直接在 test_data_path 下找图片
        img_paths = glob.glob(os.path.join(args.test_data_path, '*.png'))
    else:
        img_paths = glob.glob(os.path.join(img_dir, '*.png'))
        
    if not img_paths:
        print(f"No images found in {args.test_data_path}")
        return

    # 加载模型
    model = CNN()
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_cnn.pth')
    
    try:
        model.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 结果字典
    results = {}
    
    BATCH_SIZE = 32
    batch_imgs = []
    batch_names = [] # 记录文件名用于 JSON key
    
    # 开始推理
    for i, img_path in enumerate(img_paths):
        img = cv_imread(img_path)
        if img is None: continue
        
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        batch_imgs.append(img[np.newaxis, :, :])
        
        # 获取文件名 (不含扩展名)
        file_name = os.path.basename(img_path).replace('.png', '')
        batch_names.append(file_name)
        
        # 批处理
        if len(batch_imgs) == BATCH_SIZE or i == len(img_paths) - 1:
            batch_t = torch.tensor(np.array(batch_imgs)).float().to(DEVICE)
            
            with torch.no_grad():
                out = model.forward(batch_t)
                probs = out.cpu().numpy().flatten()
            
            # 存入结果
            for name, prob in zip(batch_names, probs):
                # JSON 要求 bool 类型: true 表示 defective, false 表示 non-defective
                # 这里使用 THRESHOLD (0.5) 进行截断
                is_defective = bool(prob > THRESHOLD)
                results[name] = is_defective
                
            batch_imgs = []
            batch_names = []

    # 保存 JSON 文件
    json_filename = f"{STUDENT_ID}.json"
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Inference done. Results saved to {json_filename}")

if __name__ == '__main__':
    inference()