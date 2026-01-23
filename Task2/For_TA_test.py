import os
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score
import glob


STUDENT_ID = "PB23111669" 


MODEL_FILENAME = "best_model.pth"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 最佳阈值 

BEST_THRESHOLDS = [0.50, 0.40, 0.35, 0.45]


IMG_SIZE = 336


class GlassDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_dir = os.path.join(root_dir, "img")
        self.txt_dir = os.path.join(root_dir, "txt")
        self.transform = transform
        # 兼容 .png 和 .PNG
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
        if len(self.img_files) == 0:
            self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.PNG")))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        file_name = os.path.basename(img_path)
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        txt_path = os.path.join(self.txt_dir, txt_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE))

        if self.transform:
            image = self.transform(image)


        label = torch.zeros(4, dtype=torch.float32)

        if not os.path.exists(txt_path):
            label[0] = 1.0 
        else:
            try:
                with open(txt_path, "r") as f:
                    lines = f.readlines()
                if len(lines) == 0:
                    label[0] = 1.0
                else:
                    has_valid_defect = False
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            cid = int(parts[0])
                            # 0->1, 1->2, 2->3
                            if 0 <= cid <= 2:
                                label[cid + 1] = 1.0
                                has_valid_defect = True
                    if not has_valid_defect:
                        label[0] = 1.0
            except:
                label[0] = 1.0 

        return image, label

def get_model():
    
    try:
        model = models.efficientnet_b3(weights=None)
        # 修改分类头
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 4)
    except AttributeError:
        
        raise ImportError("Torchvision version too old, cannot load EfficientNet.")
    
    return model

def main():

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to test dataset')
    args = parser.parse_args()

   
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = GlassDataset(args.test_data_path, transform=val_transform)
   
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)


    model = get_model()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, MODEL_FILENAME)
    
    if not os.path.exists(model_path):

        print(f"{STUDENT_ID}:0.0000") 
        return

    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception as e:

        print(f"{STUDENT_ID}:0.0000")
        return

    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:
            images = images.to(DEVICE)

            logits1 = model(images)
            probs1 = torch.sigmoid(logits1)

            probs2 = torch.sigmoid(model(torch.flip(images, [3])))

            probs3 = torch.sigmoid(model(torch.flip(images, [2])))

            avg_probs = (probs1 + probs2 + probs3) / 3.0
            avg_probs = avg_probs.cpu().numpy()

            batch_preds = np.zeros_like(avg_probs)

            for i in range(4):
                batch_preds[:, i] = (avg_probs[:, i] > BEST_THRESHOLDS[i]).astype(float)
            
            all_preds.append(batch_preds)
            all_labels.append(labels.numpy())

    if len(all_preds) > 0:
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        score = f1_score(all_labels, all_preds, average='micro')
    else:
        score = 0.0

    print(f"{STUDENT_ID}:{score:.4f}")

if __name__ == "__main__":
    main()