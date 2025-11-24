import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ============================================================
# 1. Dataset
# ============================================================
class UNetDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (960, 960))
        mask = cv2.resize(mask, (960, 960), interpolation=cv2.INTER_NEAREST)
        if self.transform:
            image = self.transform(image)
        mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0  # [1,H,W]
        return image, mask

# ============================================================
# 2. U-Net 模型（簡版）
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_c=3, out_c=1):
        super().__init__()
        self.down1 = DoubleConv(in_c, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.bottom = DoubleConv(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv4 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_c, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(self.pool(c1))
        c3 = self.down3(self.pool(c2))
        c4 = self.down4(self.pool(c3))
        c5 = self.bottom(self.pool(c4))
        x = self.up1(c5)
        x = self.conv1(torch.cat([x, c4], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, c3], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x, c2], dim=1))
        x = self.up4(x)
        x = self.conv4(torch.cat([x, c1], dim=1))
        return torch.sigmoid(self.final(x))

# ============================================================
# 3. 訓練設定
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

transform = transforms.Compose([transforms.ToTensor()])
dataset = UNetDataset("dataset/images", "dataset/masks", transform)
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

# ============================================================
# 4. 訓練迴圈
# ============================================================
if __name__ == "__main__":
    for epoch in range(1, 31):
        model.train()
        total_loss = 0
        for imgs, masks in tqdm(loader, desc=f"Epoch {epoch}"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} - Loss: {total_loss / len(loader):.4f}")
        torch.save(model.state_dict(), f"unet_epoch{epoch}.pth")
