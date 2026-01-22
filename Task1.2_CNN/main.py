import os
import argparse
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

# --- 配置 ---
IMG_SIZE = 128   # CNN 可以处理大图，保留更多细节
BATCH_SIZE = 32
LR = 0.001       # 学习率
EPOCHS = 40
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. 手动实现的卷积层 (基于 im2col 加速) ---
class ManualConv2d:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Kaiming Initialization
        n = in_channels * kernel_size * kernel_size
        stdv = 1. / np.sqrt(n)
        self.W = torch.FloatTensor(out_channels, in_channels * kernel_size * kernel_size).uniform_(-stdv, stdv).to(DEVICE)
        self.b = torch.zeros(1, out_channels).to(DEVICE)
        
        # Adam 缓存
        self.m_W, self.v_W = torch.zeros_like(self.W), torch.zeros_like(self.W)
        self.m_b, self.v_b = torch.zeros_like(self.b), torch.zeros_like(self.b)
        self.t = 0
        
        # 中间变量缓存
        self.input_shape = None
        self.x_col = None

    def forward(self, x):
        # x shape: (N, C, H, W)
        self.input_shape = x.shape
        N, C, H, W = x.shape
        
        # 计算输出尺寸
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 1. im2col: 将图片块展平成列
        # 使用 F.unfold (在白名单中) 实现高效展平
        # x_unfold shape: (N, C*K*K, H_out*W_out)
        inp_unf = F.unfold(x, (self.kernel_size, self.kernel_size), padding=self.padding, stride=self.stride)
        
        # 2. 矩阵乘法
        # W shape: (Out_C, In_C*K*K)
        # 我们需要 W @ x_col
        # 为了 Batch 计算，我们将 inp_unf 转置为: (N, H_out*W_out, C*K*K) -> 合并 Batch -> (N*H_out*W_out, C*K*K)
        # 但更简单的是：(N, C_in*K*K, L) -> permute -> (N, L, C_in*K*K)
        
        self.x_col = inp_unf # 缓存用于反向传播 (N, Cin*K*K, L)
        
        # 执行卷积: Out = W * X_col + b
        # (Out_C, Cin*K*K) @ (N, Cin*K*K, L) -> 这一步需要广播
        # 我们可以利用 transpose 做： (N, L, Cin*K*K) @ W.t() -> (N, L, Out_C)
        
        out = inp_unf.transpose(1, 2).matmul(self.W.t()) # (N, L, Out_C)
        out += self.b # 广播加偏置
        
        # Reshape 回图片格式 (N, Out_C, H_out, W_out)
        out = out.transpose(1, 2).view(N, self.out_channels, H_out, W_out)
        return out

    def backward(self, grad_output):
        # grad_output: (N, Out_C, H_out, W_out)
        N, Out_C, H_out, W_out = grad_output.shape
        
        # 1. 准备梯度
        # Reshape grad: (N, Out_C, L) where L = H_out*W_out
        grad_out_flat = grad_output.view(N, Out_C, -1)
        
        # 2. 计算 dW
        # dW = grad_out * x_col^T
        # Sum over batch: (Out_C, L) @ (L, Cin*K*K) ??? No.
        # 正确公式: dW = sum_over_batch( grad_out_n @ x_col_n.T )
        # grad_out_flat: (N, Out_C, L)
        # x_col: (N, Cin*K*K, L)
        # result: (N, Out_C, Cin*K*K) -> sum(0)
        
        self.dW = grad_out_flat.matmul(self.x_col.transpose(1, 2)).sum(dim=0)
        self.db = grad_out_flat.sum(dim=(0, 2)).unsqueeze(0) # sum over N and L
        
        # 3. 计算 dx (输入梯度)
        # dx_col = W^T * grad_out
        # (Cin*K*K, Out_C) @ (N, Out_C, L) -> (N, Cin*K*K, L)
        grad_x_col = self.W.t().matmul(grad_out_flat)
        
        # 4. col2im: 将列恢复为图片梯度
        # 使用 F.fold
        grad_x = F.fold(grad_x_col, output_size=(self.input_shape[2], self.input_shape[3]), 
                        kernel_size=(self.kernel_size, self.kernel_size), 
                        padding=self.padding, stride=self.stride)
        
        return grad_x

    def step(self, lr):
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        # Adam update for W
        self.m_W = beta1 * self.m_W + (1 - beta1) * self.dW
        self.v_W = beta2 * self.v_W + (1 - beta2) * (self.dW**2)
        m_hat = self.m_W / (1 - beta1**self.t)
        v_hat = self.v_W / (1 - beta2**self.t)
        self.W -= lr * m_hat / (torch.sqrt(v_hat) + eps)
        
        # Adam update for b
        self.m_b = beta1 * self.m_b + (1 - beta1) * self.db
        self.v_b = beta2 * self.v_b + (1 - beta2) * (self.db**2)
        m_hat_b = self.m_b / (1 - beta1**self.t)
        v_hat_b = self.v_b / (1 - beta2**self.t)
        self.b -= lr * m_hat_b / (torch.sqrt(v_hat_b) + eps)

# --- 2. 手动池化层 ---
class ManualMaxPool2d:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.indices = None
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        # 使用 return_indices=True 来辅助反向传播
        out, indices = F.max_pool2d(x, self.kernel_size, self.stride, return_indices=True)
        self.indices = indices
        return out

    def backward(self, grad_output):
        # 使用 F.max_unpool2d 利用之前记录的 indices 恢复梯度
        grad_input = F.max_unpool2d(grad_output, self.indices, self.kernel_size, self.stride, output_size=self.input_shape)
        return grad_input

# --- 3. 辅助层 ---
class ManualFlatten:
    def __init__(self):
        self.input_shape = None
    def forward(self, x):
        self.input_shape = x.shape
        return x.view(x.shape[0], -1)
    def backward(self, grad_output):
        return grad_output.view(self.input_shape)

class ManualReLU:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x > 0).float()
        return x * self.mask
    def backward(self, grad_output):
        return grad_output * self.mask

class ManualLinear: # (和之前一样，为了完整性保留)
    def __init__(self, in_features, out_features):
        std = np.sqrt(2.0 / in_features)
        self.W = torch.randn(in_features, out_features).to(DEVICE) * std
        self.b = torch.zeros(1, out_features).to(DEVICE)
        self.m_W, self.v_W = torch.zeros_like(self.W), torch.zeros_like(self.W)
        self.m_b, self.v_b = torch.zeros_like(self.b), torch.zeros_like(self.b)
        self.t = 0
        self.input = None
    def forward(self, x):
        self.input = x
        return x.mm(self.W) + self.b
    def backward(self, grad_output):
        self.dW = self.input.t().mm(grad_output)
        self.db = grad_output.sum(dim=0, keepdim=True)
        return grad_output.mm(self.W.t())
    def step(self, lr):
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.m_W = beta1 * self.m_W + (1 - beta1) * self.dW
        self.v_W = beta2 * self.v_W + (1 - beta2) * (self.dW**2)
        m_hat = self.m_W / (1 - beta1**self.t)
        v_hat = self.v_W / (1 - beta2**self.t)
        self.W -= lr * m_hat / (torch.sqrt(v_hat) + eps)
        self.m_b = beta1 * self.m_b + (1 - beta1) * self.db
        self.v_b = beta2 * self.v_b + (1 - beta2) * (self.db**2)
        m_hat_b = self.m_b / (1 - beta1**self.t)
        v_hat_b = self.v_b / (1 - beta2**self.t)
        self.b -= lr * m_hat_b / (torch.sqrt(v_hat_b) + eps)

class ManualSigmoid:
    def __init__(self):
        self.out = None
    def forward(self, x):
        self.out = 1.0 / (1.0 + torch.exp(-torch.clamp(x, -50, 50)))
        return self.out
    def backward(self, grad_output):
        return grad_output * self.out * (1.0 - self.out)

# --- 4. CNN 模型整合 ---
class CNN:
    def __init__(self):
        # 输入: (1, 128, 128)
        self.layers = [
            # Conv1: 1 -> 8, 5x5
            ManualConv2d(1, 8, kernel_size=5, stride=1, padding=2), # (8, 128, 128)
            ManualReLU(),
            ManualMaxPool2d(2, 2), # (8, 64, 64)
            
            # Conv2: 8 -> 16, 3x3
            ManualConv2d(8, 16, kernel_size=3, stride=1, padding=1), # (16, 64, 64)
            ManualReLU(),
            ManualMaxPool2d(2, 2), # (16, 32, 32)
            
            # Conv3: 16 -> 32, 3x3
            ManualConv2d(16, 32, kernel_size=3, stride=1, padding=1), # (32, 32, 32)
            ManualReLU(),
            ManualMaxPool2d(2, 2), # (32, 16, 16)
            
            ManualFlatten(),
            # Linear: 32 * 16 * 16 = 8192
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

    def backward(self, grad_loss):
        grad = grad_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, lr):
        for layer in self.layers:
            if hasattr(layer, 'step'):
                layer.step(lr)
                
    def save(self, path):
        # 保存所有 ManualConv2d 和 ManualLinear 的参数
        checkpoint = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (ManualConv2d, ManualLinear)):
                checkpoint[f'{i}_W'] = layer.W.cpu()
                checkpoint[f'{i}_b'] = layer.b.cpu()
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (ManualConv2d, ManualLinear)):
                layer.W = checkpoint[f'{i}_W'].to(DEVICE)
                layer.b = checkpoint[f'{i}_b'].to(DEVICE)

# --- 5. 数据加载 (解决类别不平衡) ---
def get_balanced_batch_cnn(img_paths, txt_dir, batch_size):
    # 实时读取图片，不占内存
    # 随机选择一半正样本，一半负样本
    
    # 预先扫描所有样本的标签（只做一次）
    if not hasattr(get_balanced_batch_cnn, 'pos_paths'):
        pos_paths = []
        neg_paths = []
        for p in img_paths:
            base = os.path.basename(p).replace('.png', '')
            if os.path.exists(os.path.join(txt_dir, base + '.txt')):
                pos_paths.append(p)
            else:
                neg_paths.append(p)
        get_balanced_batch_cnn.pos_paths = pos_paths
        get_balanced_batch_cnn.neg_paths = neg_paths
        print(f"Dataset stats: {len(pos_paths)} Defective, {len(neg_paths)} Good")

    pos_batch_paths = np.random.choice(get_balanced_batch_cnn.pos_paths, batch_size // 2)
    neg_batch_paths = np.random.choice(get_balanced_batch_cnn.neg_paths, batch_size - (batch_size // 2))
    batch_paths = np.concatenate([pos_batch_paths, neg_batch_paths])
    np.random.shuffle(batch_paths)

    images = []
    labels = []
    
    for p in batch_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        # CNN 输入需要 (C, H, W)
        images.append(img[np.newaxis, :, :]) 
        
        base = os.path.basename(p).replace('.png', '')
        label = 1.0 if os.path.exists(os.path.join(txt_dir, base + '.txt')) else 0.0
        labels.append(label)
        
    return torch.tensor(np.array(images)).float().to(DEVICE), \
           torch.tensor(np.array(labels)).float().view(-1, 1).to(DEVICE)

# --- 6. 主训练循环 ---
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../dataset/train')
    args = parser.parse_args()
    
    img_dir = os.path.join(args.data_path, 'img')
    txt_dir = os.path.join(args.data_path, 'txt')
    all_img_paths = glob.glob(os.path.join(img_dir, '*.png'))
    
    # 初始化
    model = CNN()
    print(f"Model initialized on {DEVICE}. Training CNN...")
    
    # 简单的 Train/Val split (取最后20%做验证)
    split = int(0.8 * len(all_img_paths))
    train_paths = all_img_paths[:split]
    val_paths = all_img_paths[split:]
    
    best_f1 = 0.0
    steps = len(train_paths) // BATCH_SIZE
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        # Training
        for _ in range(steps):
            bx, by = get_balanced_batch_cnn(train_paths, txt_dir, BATCH_SIZE)
            
            # Forward
            out = model.forward(bx)
            
            # Loss (BCE)
            out = torch.clamp(out, 1e-7, 1 - 1e-7)
            loss = - (by * torch.log(out) + (1 - by) * torch.log(1 - out)).mean()
            
            # Backward
            grad_out = - (by / out - (1 - by) / (1 - out)) / BATCH_SIZE
            model.backward(grad_out)
            model.step(LR)
            
            total_loss += loss.item()
            
        # Validation (Sampled to save time)
        val_steps = 10 
        val_preds, val_targets = [], []
        
        for _ in range(val_steps):
            # 验证集也用 balanced batch 来查看 F1 能力
            bx, by = get_balanced_batch_cnn(val_paths, txt_dir, BATCH_SIZE)
            with torch.no_grad(): # Manual no grad logic implies just dont call backward
                out = model.forward(bx)
                val_preds.extend(out.cpu().numpy().flatten())
                val_targets.extend(by.cpu().numpy().flatten())
                
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        # 寻找最佳阈值
        cur_best_f1 = 0
        best_thresh = 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            p_bin = (val_preds > t).astype(float)
            f1 = f1_score(val_targets, p_bin)
            if f1 > cur_best_f1:
                cur_best_f1 = f1
                best_thresh = t
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/steps:.4f} | Val F1: {cur_best_f1:.4f} @ {best_thresh:.2f}")
        
        if cur_best_f1 > best_f1:
            best_f1 = cur_best_f1
            model.save('model_cnn.pth')
            print("  >>> Model Saved")

if __name__ == '__main__':
    train()