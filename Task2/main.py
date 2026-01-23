import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import random

DATAROOT = "../dataset"
TRAIN_IMG_DIR = os.path.join(DATAROOT, "train/img")
TRAIN_TXT_DIR = os.path.join(DATAROOT, "train/txt")
TEST_IMG_DIR = os.path.join(DATAROOT, "test/img")
TEST_TXT_DIR = os.path.join(DATAROOT, "test/txt")

IMG_SIZE = 336  
BATCH_SIZE = 64 
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
NUM_WORKERS = 16 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "best_model.pth"
NUM_CLASSES = 4
scaler = GradScaler()


CLASS_WEIGHTS = [1.0, 2.0, 5.0, 5.0] 

class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        probas = torch.sigmoid(inputs)
        

        p_t = targets * probas + (1 - targets) * (1 - probas)
        focal_factor = (1 - p_t) ** self.gamma
        loss = bce_loss * focal_factor


        if self.pos_weight is not None:

            w = self.pos_weight.to(inputs.device)
            loss = loss * w 
        
        return loss.mean()


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, img):
        return img + torch.randn(img.size()) * self.std + self.mean

class GlassDataset(Dataset):
    def __init__(self, img_dir, txt_dir, transform=None):
        self.img_dir = img_dir
        self.txt_dir = txt_dir
        self.transform = transform
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        file_name = os.path.basename(img_path)
        txt_name = file_name.replace(".png", ".txt")
        txt_path = os.path.join(self.txt_dir, txt_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.zeros(NUM_CLASSES, dtype=torch.float32)

        if not os.path.exists(txt_path):
            label[0] = 1.0
        else:
            with open(txt_path, "r") as f:
                lines = f.readlines()
            if len(lines) == 0:
                label[0] = 1.0
            else:
                has_defect = False
                for line in lines:
                    try:
                        class_id = int(line.strip().split()[0])
                        if 0 <= class_id <= 2:
                            label[class_id + 1] = 1.0
                            has_defect = True
                    except:
                        continue
                if not has_defect:
                    label[0] = 1.0
        return image, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),

        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5), 
        transforms.RandomAutocontrast(p=0.5),
        
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.RandomApply([GaussianNoise(0., 0.05)], p=0.3),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def evaluate_with_tta(model, loader, device, return_raw=False):
    model.eval()
    all_probs = []
    all_labels = []

    print("Evaluating with TTA (Original + Flip H + Flip V)...")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="TTA Eval"):
            images = images.to(device)
            

            logits1 = model(images)
            probs1 = torch.sigmoid(logits1)
            

            probs2 = torch.sigmoid(model(torch.flip(images, [3])))

            probs3 = torch.sigmoid(model(torch.flip(images, [2])))

            avg_probs = (probs1 + probs2 + probs3) / 3.0
            
            all_probs.append(avg_probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    if return_raw:
        return all_probs, all_labels
        
    return 0.0 

def find_best_threshold_per_class(probs, labels):
    """为每个类别单独寻找最佳阈值"""
    best_thresholds = []
    best_f1s = []
    

    class_names = ["No Defect", "Chipped", "Scratch", "Stain"]
    
    print("\n--- Searching Best Thresholds ---")
    for i in range(NUM_CLASSES):
        best_f1 = 0
        best_t = 0.5

        for t in np.arange(0.1, 0.95, 0.05):
            preds = (probs[:, i] > t).astype(float)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        
        best_thresholds.append(best_t)
        best_f1s.append(best_f1)
        print(f"Class {i} ({class_names[i]}): Best T={best_t:.2f}, F1={best_f1:.4f}")

    final_preds = np.zeros_like(probs)
    for i in range(NUM_CLASSES):
        final_preds[:, i] = (probs[:, i] > best_thresholds[i]).astype(float)
    
    micro_f1 = f1_score(labels, final_preds, average='micro')
    return best_thresholds, micro_f1

def main():
    print(f"Using device: {DEVICE}")
    train_transform, val_transform = get_transforms()
    
    train_dataset = GlassDataset(TRAIN_IMG_DIR, TRAIN_TXT_DIR, transform=train_transform)
    test_dataset = GlassDataset(TEST_IMG_DIR, TEST_TXT_DIR, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    try:
        print("Loading EfficientNet B3...")
        weights = models.EfficientNet_B3_Weights.DEFAULT
        model = models.efficientnet_b3(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    except:
        print("EfficientNet failed, fallback to ResNet50")
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    model = model.to(DEVICE)

    cls_weights = torch.tensor(CLASS_WEIGHTS).float()
    criterion = MultiLabelFocalLoss(alpha=0.25, gamma=2, pos_weight=cls_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_f1 = 0.0
    best_thresholds = [0.5] * NUM_CLASSES

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        probs, labels = evaluate_with_tta(model, test_loader, DEVICE, return_raw=True)
        curr_thresholds, curr_f1 = find_best_threshold_per_class(probs, labels)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Micro F1 (Optimized): {curr_f1:.4f}")

        if curr_f1 > best_val_f1:
            best_val_f1 = curr_f1
            best_thresholds = curr_thresholds
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"--> Saved Best Model! (F1: {best_val_f1:.4f})")
            print(f"--> Best Thresholds saved: {best_thresholds}")

    print(f"\nTraining Complete. Best Micro F1: {best_val_f1:.4f}")
    print(f"Final Inference Thresholds: {best_thresholds}")

if __name__ == "__main__":
    main()