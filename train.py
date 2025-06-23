import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from tqdm import tqdm

from models import AutoDriveNet
from datasets import AutoDriveDataset

# 初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoDriveNet().to(device)
writer = SummaryWriter(log_dir="runs/AutoDriveNet")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((120, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
train_set = AutoDriveDataset(data_folder="data", transform=transform, mode="train")
val_set = AutoDriveDataset(data_folder="data", transform=transform, mode="val")
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# 损失函数、优化器、调度器
criterion = nn.SmoothL1Loss()  # Huber Loss
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

best_val_loss = float('inf')
patience, counter = 5, 0
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    train_loss, train_preds, train_targets = 0, [], []

    for imgs, angles in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]"):
        imgs, angles = imgs.to(device), angles.to(device)
        outputs = model(imgs)

        loss = criterion(outputs, angles)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)
        train_preds.extend(outputs.cpu().detach().numpy())
        train_targets.extend(angles.cpu().numpy())

    train_loss /= len(train_loader.dataset)
    train_mae = mean_absolute_error(train_targets, train_preds)
    train_mse = mean_squared_error(train_targets, train_preds)
    train_rmse = sqrt(train_mse)

    # 验证阶段
    model.eval()
    val_loss, val_preds, val_targets = 0, [], []

    with torch.no_grad():
        for imgs, angles in val_loader:
            imgs, angles = imgs.to(device), angles.to(device)
            outputs = model(imgs)

            loss = criterion(outputs, angles)
            val_loss += loss.item() * imgs.size(0)
            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(angles.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_mae = mean_absolute_error(val_targets, val_preds)
    val_mse = mean_squared_error(val_targets, val_preds)
    val_rmse = sqrt(val_mse)

    # TensorBoard 记录
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Val", val_loss, epoch)
    writer.add_scalar("MAE/Train", train_mae, epoch)
    writer.add_scalar("MAE/Val", val_mae, epoch)
    writer.add_scalar("MSE/Train", train_mse, epoch)
    writer.add_scalar("MSE/Val", val_mse, epoch)
    writer.add_scalar("RMSE/Train", train_rmse, epoch)
    writer.add_scalar("RMSE/Val", val_rmse, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

    print(f"\nEpoch {epoch+1}:")
    print(f"Train Loss={train_loss:.4f}, MAE={train_mae:.4f}, RMSE={train_rmse:.4f}")
    print(f"Val   Loss={val_loss:.4f}, MAE={val_mae:.4f}, RMSE={val_rmse:.4f}")

    # 更新学习率调度器
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✅ 模型已保存，验证损失改进为 {val_loss:.4f}")
    else:
        counter += 1
        print(f"⚠️ 验证损失未改进 ({counter}/{patience})")

    if counter >= patience:
        print("⛔ 早停触发，训练终止。")
        break

writer.close()
