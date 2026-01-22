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
IMG_SIZE = 128
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# 1. 基础算子 (保持不变)
# ==============================================================================
class ManualConv2d:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
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
# 2. CNN 模型
# ==============================================================================
class CNN:
    def __init__(self):
        self.layers = [
            ManualConv2d(1, 8, 5, 1, 2), ManualReLU(), ManualMaxPool2d(2, 2),
            ManualConv2d(8, 16, 3, 1, 1), ManualReLU(), ManualMaxPool2d(2, 2),
            ManualConv2d(16, 32, 3, 1, 1), ManualReLU(), ManualMaxPool2d(2, 2),
            ManualFlatten(),
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
# 3. 两个不同的加载器 (核心修改！)
# ==============================================================================

class TrainLoader:
    """训练专用：平衡采样 (50/50) + 数据增强"""
    def __init__(self, paths, txt_dir):
        self.pos = []
        self.neg = []
        for p in paths:
            base = os.path.basename(p).replace('.png', '')
            if os.path.exists(os.path.join(txt_dir, base + '.txt')): self.pos.append(p)
            else: self.neg.append(p)
        print(f"[Train] Balanced Loader: {len(self.pos)} Pos, {len(self.neg)} Neg")

    def get_batch(self, batch_size):
        # 强制平衡：一半正，一半负
        n_pos = batch_size // 2
        batch_paths = np.concatenate([
            np.random.choice(self.pos, n_pos),
            np.random.choice(self.neg, batch_size - n_pos)
        ])
        np.random.shuffle(batch_paths)
        
        imgs, lbls = [], []
        for p in batch_paths:
            raw = np.fromfile(p, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
            
            # 增强 (Flip/Rotate)
            if np.random.rand() > 0.5: img = np.flip(img, axis=1)
            if np.random.rand() > 0.5: img = np.flip(img, axis=0)
            k = np.random.randint(0, 4)
            if k > 0: img = np.rot90(img, k)
            
            imgs.append(img[np.newaxis, :, :].copy())
            base = os.path.basename(p).replace('.png', '')
            # 这里不用再查文件系统，因为 init 里分过了，但为了稳健还是查一下
            # 简化逻辑：如果在 pos 列表里就是 1
            lbl = 1.0 if p in self.pos else 0.0
            lbls.append(lbl)
            
        return torch.tensor(np.array(imgs)).float().to(DEVICE), \
               torch.tensor(np.array(lbls)).float().view(-1, 1).to(DEVICE)

class ValLoader:
    """验证专用：不平衡 (真实分布) + 无增强 + 顺序读取"""
    def __init__(self, paths, txt_dir):
        self.paths = paths
        self.txt_dir = txt_dir
        self.n = len(paths)
        # 统计一下真实比例
        pos_cnt = sum([1 for p in paths if os.path.exists(os.path.join(txt_dir, os.path.basename(p).replace('.png','.txt')))])
        print(f"[Val] Real Distribution: {pos_cnt} Pos, {self.n - pos_cnt} Neg (Ratio 1:{ (self.n-pos_cnt)/pos_cnt if pos_cnt>0 else 'inf' })")

    def get_all_batches(self, batch_size):
        # 生成器，按顺序吐出所有数据
        for i in range(0, self.n, batch_size):
            batch_paths = self.paths[i : i + batch_size]
            imgs, lbls = [], []
            for p in batch_paths:
                raw = np.fromfile(p, dtype=np.uint8)
                img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
                # 无增强！
                imgs.append(img[np.newaxis, :, :].copy())
                
                base = os.path.basename(p).replace('.png', '')
                lbl = 1.0 if os.path.exists(os.path.join(self.txt_dir, base + '.txt')) else 0.0
                lbls.append(lbl)
            
            yield torch.tensor(np.array(imgs)).float().to(DEVICE), \
                  torch.tensor(np.array(lbls)).float().view(-1, 1).to(DEVICE)

# ==============================================================================
# 4. 训练循环
# ==============================================================================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../dataset/train')
    args = parser.parse_args()
    
    img_dir = os.path.join(args.data_path, 'img')
    txt_dir = os.path.join(args.data_path, 'txt')
    all_paths = glob.glob(os.path.join(img_dir, '*.png'))
    if not all_paths: print("No images!"); return

    # 1. 物理切分 (Train 80%, Val 20%)
    np.random.seed(42) # 固定种子，保证可复现
    np.random.shuffle(all_paths)
    split = int(0.8 * len(all_paths))
    train_paths = all_paths[:split]
    val_paths = all_paths[split:]

    # 2. 初始化不同的 Loader
    train_loader = TrainLoader(train_paths, txt_dir) # 训练用：平衡+增强
    val_loader = ValLoader(val_paths, txt_dir)       # 验证用：真实分布

    model = CNN()
    print(f"Start Training on {DEVICE}...")
    
    best_f1 = 0.0
    patience, pat_cnt = 8, 0
    lr, lr_pat, lr_cnt = LR, 3, 0
    history = {'loss': [], 'val_f1': []}

    for epoch in range(EPOCHS):
        # --- 训练 ---
        steps = len(train_paths) // BATCH_SIZE
        loss_sum = 0
        for _ in range(steps):
            bx, by = train_loader.get_batch(BATCH_SIZE)
            out = model.forward(bx)
            out = torch.clamp(out, 1e-7, 1-1e-7)
            loss = -(by * torch.log(out) + (1-by) * torch.log(1-out)).mean()
            model.backward(-(by/out - (1-by)/(1-out))/BATCH_SIZE)
            model.step(lr)
            loss_sum += loss.item()
        
        avg_loss = loss_sum / steps
        history['loss'].append(avg_loss)

        # --- 验证 (全量扫描，寻找真实的最佳阈值) ---
        all_probs, all_targets = [], []
        # 使用 yield 遍历整个验证集
        for bx, by in val_loader.get_all_batches(BATCH_SIZE):
            with torch.no_grad():
                out = model.forward(bx)
                all_probs.extend(out.cpu().numpy().flatten())
                all_targets.extend(by.cpu().numpy().flatten())
        
        # 搜索最佳阈值 (这才是真实的 F1)
        vp, vt = np.array(all_probs), np.array(all_targets)
        cur_best_f1, best_t = 0.0, 0.5
        
        # 因为验证集很不平衡，阈值通常很低，多搜一下低分段
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (vp > t).astype(int)
            s = f1_score(vt, preds, zero_division=0)
            if s > cur_best_f1:
                cur_best_f1 = s
                best_t = t
        
        history['val_f1'].append(cur_best_f1)
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val F1: {cur_best_f1:.4f} @ {best_t:.2f} | LR: {lr:.5f}")

        # 策略更新
        if cur_best_f1 > best_f1:
            best_f1, pat_cnt, lr_cnt = cur_best_f1, 0, 0
            model.save('model_cnn.pth')
            print(f"  >>> Best Saved: {best_f1:.4f}")
        else:
            pat_cnt += 1; lr_cnt += 1
            if lr_cnt >= lr_pat: lr *= 0.5; lr_cnt = 0; print(f"  vvv LR Decay: {lr}")
            if pat_cnt >= patience: print("Early Stop"); break

    # 绘图
    plt.figure(); plt.plot(history['loss']); plt.savefig('loss.png')
    plt.figure(); plt.plot(history['val_f1']); plt.savefig('f1.png')

if __name__ == '__main__':
    train()