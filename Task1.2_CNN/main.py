import os
import argparse
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

# --- 配置 ---
IMG_SIZE = 128   # 保持128平衡速度与精度
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# 1. 基础算子 (无 Autograd)
# ==============================================================================
class ManualConv2d:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # He Initialization
        n = in_channels * kernel_size * kernel_size
        stdv = np.sqrt(2. / n)
        self.W = torch.randn(out_channels, n).to(DEVICE) * stdv
        self.b = torch.zeros(1, out_channels).to(DEVICE)
        self.m_W, self.v_W = torch.zeros_like(self.W), torch.zeros_like(self.W)
        self.m_b, self.v_b = torch.zeros_like(self.b), torch.zeros_like(self.b)
        self.t = 0

    def forward(self, x):
        self.input_shape = x.shape
        N, C, H, W = x.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        inp_unf = F.unfold(x, (self.kernel_size, self.kernel_size), padding=self.padding, stride=self.stride)
        self.x_col = inp_unf 
        out = inp_unf.transpose(1, 2).matmul(self.W.t()) + self.b 
        out = out.transpose(1, 2).reshape(N, self.out_channels, H_out, W_out)
        return out

    def backward(self, grad_output):
        N, Out_C, H_out, W_out = grad_output.shape
        grad_out_flat = grad_output.reshape(N, Out_C, -1)
        self.dW = grad_out_flat.matmul(self.x_col.transpose(1, 2)).sum(dim=0)
        self.db = grad_out_flat.sum(dim=(0, 2)).unsqueeze(0)
        grad_x_col = self.W.t().matmul(grad_out_flat)
        grad_x = F.fold(grad_x_col, output_size=(self.input_shape[2], self.input_shape[3]), 
                        kernel_size=(self.kernel_size, self.kernel_size), 
                        padding=self.padding, stride=self.stride)
        return grad_x

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

class ManualMaxPool2d:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, x):
        self.input_shape = x.shape
        out, self.indices = F.max_pool2d(x, self.kernel_size, self.stride, return_indices=True)
        return out
    def backward(self, grad_output):
        return F.max_unpool2d(grad_output, self.indices, self.kernel_size, self.stride, output_size=self.input_shape)

class ManualFlatten:
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1) 
    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

class ManualReLU:
    def forward(self, x):
        self.mask = (x > 0).float()
        return x * self.mask
    def backward(self, grad_output):
        return grad_output * self.mask

class ManualLinear:
    def __init__(self, in_features, out_features):
        std = np.sqrt(2.0 / in_features)
        self.W = torch.randn(in_features, out_features).to(DEVICE) * std
        self.b = torch.zeros(1, out_features).to(DEVICE)
        self.m_W, self.v_W = torch.zeros_like(self.W), torch.zeros_like(self.W)
        self.m_b, self.v_b = torch.zeros_like(self.b), torch.zeros_like(self.b)
        self.t = 0
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
    def forward(self, x):
        self.out = 1.0 / (1.0 + torch.exp(-torch.clamp(x, -50, 50)))
        return self.out
    def backward(self, grad_output):
        return grad_output * self.out * (1.0 - self.out)

# ==============================================================================
# 2. 轻量级 CNN (3层卷积)
# ==============================================================================
class CNN:
    def __init__(self):
        self.layers = [
            # Conv1: 1 -> 8
            ManualConv2d(1, 8, kernel_size=5, stride=1, padding=2), ManualReLU(), ManualMaxPool2d(2, 2),
            # Conv2: 8 -> 16
            ManualConv2d(8, 16, kernel_size=3, stride=1, padding=1), ManualReLU(), ManualMaxPool2d(2, 2),
            # Conv3: 16 -> 32
            ManualConv2d(16, 32, kernel_size=3, stride=1, padding=1), ManualReLU(), ManualMaxPool2d(2, 2),
            
            ManualFlatten(),
            # Linear: 32 * 16 * 16 = 8192 (For 128x128 input)
            ManualLinear(32 * 16 * 16, 128), ManualReLU(),
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
        checkpoint = torch.load(path, map_location=DEVICE)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (ManualConv2d, ManualLinear)):
                layer.W = checkpoint[f'{i}_W'].to(DEVICE); layer.b = checkpoint[f'{i}_b'].to(DEVICE)

# ==============================================================================
# 3. 安全的数据加载器 (Class-based, Zero Leakage)
# ==============================================================================
class GlassDataLoader:
    def __init__(self, img_paths, txt_dir, name="Loader"):
        """
        初始化时就完成数据扫描和分类，保证训练集和验证集互不干扰。
        """
        self.name = name
        self.pos_paths = []
        self.neg_paths = []
        
        # 扫描路径
        for p in img_paths:
            base = os.path.basename(p).replace('.png', '')
            if os.path.exists(os.path.join(txt_dir, base + '.txt')):
                self.pos_paths.append(p)
            else:
                self.neg_paths.append(p)
        
        print(f"[{self.name}] Stats: {len(self.pos_paths)} Defective, {len(self.neg_paths)} Good")

    def get_batch(self, batch_size, augment=False):
        """
        获取一个平衡采样的 Batch。
        augment=True: 训练模式，启用随机翻转/旋转
        augment=False: 验证模式，禁用增强，只做缩放和归一化
        """
        # 平衡采样: 50% 正样本, 50% 负样本
        n_pos = batch_size // 2
        n_neg = batch_size - n_pos
        
        # 随机抽取路径
        # 注意：如果是验证集，这里虽然是随机抽，但因为augment=False，
        # 我们评估的是模型对这批样本的“原始形态”的判断能力，这是合理的。
        pos_batch = np.random.choice(self.pos_paths, n_pos)
        neg_batch = np.random.choice(self.neg_paths, n_neg)
        batch_paths = np.concatenate([pos_batch, neg_batch])
        np.random.shuffle(batch_paths)

        images, labels = [], []
        for p in batch_paths:
            # 读取
            raw = np.fromfile(p, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            
            # --- 数据增强 (仅当 augment=True 时触发) ---
            if augment:
                # 随机水平翻转
                if np.random.rand() > 0.5:
                    img = np.flip(img, axis=1)
                # 随机垂直翻转
                if np.random.rand() > 0.5:
                    img = np.flip(img, axis=0)
                # 随机旋转 (90度倍数)
                k = np.random.randint(0, 4)
                if k > 0:
                    img = np.rot90(img, k)
            
            # 解决内存不连续警告
            images.append(img[np.newaxis, :, :].copy()) 
            
            # 标签
            # 实际上我们在init里已经分好类了，可以直接判断 p 在不在 pos_paths 里，
            # 但为了简单，这里复用路径判断
            if p in self.pos_paths:
                labels.append(1.0)
            else:
                labels.append(0.0)
            
        return torch.tensor(np.array(images)).float().to(DEVICE), \
               torch.tensor(np.array(labels)).float().view(-1, 1).to(DEVICE)

# ==============================================================================
# 4. 训练循环 (含绘图)
# ==============================================================================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../dataset/train')
    args = parser.parse_args()
    
    # 路径检查
    img_dir = os.path.join(args.data_path, 'img')
    txt_dir = os.path.join(args.data_path, 'txt')
    all_paths = glob.glob(os.path.join(img_dir, '*.png'))
    if not all_paths: print("Error: No images found!"); return

    # --- 1. 严格切分数据集 (先切分，再给 Loader) ---
    np.random.shuffle(all_paths) # 打乱顺序
    split = int(0.8 * len(all_paths))
    train_paths = all_paths[:split]
    val_paths = all_paths[split:]
    
    print(f"Total images: {len(all_paths)}. Split: {len(train_paths)} Train, {len(val_paths)} Val.")

    # --- 2. 初始化完全独立的 Loader ---
    train_loader = GlassDataLoader(train_paths, txt_dir, name="Train")
    val_loader = GlassDataLoader(val_paths, txt_dir, name="Val")
    
    model = CNN()
    print(f"Start Training (Safe Split & Augmentation) on {DEVICE}...")
    
    # 策略参数
    best_f1 = 0.0
    patience, pat_cnt = 8, 0
    lr, lr_pat, lr_cnt = LR, 3, 0
    
    history = {'loss': [], 'val_f1': [], 'val_precision': [], 'val_recall': []}
    
    for epoch in range(EPOCHS):
        # --- 训练阶段 (augment=True) ---
        steps = len(train_paths) // BATCH_SIZE
        loss_sum = 0
        for _ in range(steps):
            bx, by = train_loader.get_batch(BATCH_SIZE, augment=True) # 开启增强
            out = model.forward(bx)
            out = torch.clamp(out, 1e-7, 1-1e-7)
            loss = -(by * torch.log(out) + (1-by) * torch.log(1-out)).mean()
            
            model.backward(-(by/out - (1-by)/(1-out))/BATCH_SIZE)
            model.step(lr)
            loss_sum += loss.item()

        avg_loss = loss_sum / steps
        history['loss'].append(avg_loss)

        # --- 验证阶段 (augment=False) ---
        # 严格禁止验证集增强，保证指标公平性
        val_preds, val_targets = [], []
        for _ in range(20): # 抽样验证
            bx, by = val_loader.get_batch(BATCH_SIZE, augment=False) # 关闭增强
            with torch.no_grad():
                out = model.forward(bx)
                val_preds.extend(out.cpu().numpy().flatten())
                val_targets.extend(by.cpu().numpy().flatten())
        
        # 找最佳阈值
        vp, vt = np.array(val_preds), np.array(val_targets)
        cur_best_f1 = 0
        cur_p, cur_r = 0, 0
        best_t = 0.5
        
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (vp > t).astype(int)
            s = f1_score(vt, preds, zero_division=0)
            if s > cur_best_f1:
                cur_best_f1 = s
                best_t = t
                cur_p = precision_score(vt, preds, zero_division=0)
                cur_r = recall_score(vt, preds, zero_division=0)
        
        history['val_f1'].append(cur_best_f1)
        history['val_precision'].append(cur_p)
        history['val_recall'].append(cur_r)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val F1: {cur_best_f1:.4f} (P:{cur_p:.2f} R:{cur_r:.2f}) | LR: {lr:.5f}")

        # 策略更新
        if cur_best_f1 > best_f1:
            best_f1, pat_cnt, lr_cnt = cur_best_f1, 0, 0
            model.save('model_cnn.pth')
            print(f"  >>> Best Saved: {best_f1:.4f}")
        else:
            pat_cnt += 1; lr_cnt += 1
            if lr_cnt >= lr_pat: lr *= 0.5; lr_cnt = 0; print(f"  vvv LR Decay: {lr}")
            if pat_cnt >= patience: print("Early Stop"); break

    # --- 绘图 ---
    print("Generating plots...")
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss', color='red')
    plt.savefig('training_loss.png')
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['val_f1'], label='Val F1', color='blue')
    plt.plot(history['val_precision'], label='Precision', linestyle='--')
    plt.plot(history['val_recall'], label='Recall', linestyle='--')
    plt.legend()
    plt.savefig('validation_metrics.png')
    print("Plots saved.")

if __name__ == '__main__':
    train()