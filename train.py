import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import numpy as np

# Data Preparation
# 1.把npy读到内存里，变成numpy数组
image_train_numpy = np.load(r'processed_phononic_9000\train_images.npy')
label_train_numpy = np.load(r'processed_phononic_9000\train_labels.npy')
# 2.把numpy数组变成变成tensor
image_train_tensor = torch.tensor(image_train_numpy, dtype=torch.float32)
label_train_tensor = torch.tensor(label_train_numpy, dtype=torch.float32)
# 3.把两个tensor合成一个dataset(Dict字典，前索引，后label)
train_set = TensorDataset(image_train_tensor, label_train_tensor)
# 4.把dataloader负责把dataset分成一个个小batch，分批加载进trinloader
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# 检测模型是否
# -----------------------------
# 验货环节：拦截第一个 Batch 的数据
# -----------------------------

# 1. 从传送带 (train_loader) 上取下第一辆卡车的货物
batch_images, batch_labels = next(iter(train_loader))

# 2. 打印看看这车货到底长啥样
print("=== 第一批次验货报告 ===")
print(f"装载的图片批次维度: {batch_images.shape}")
print(f"装载的标签批次维度: {batch_labels.shape}")
print(f"图片的数据类型: {batch_images.dtype}")
print(f"标签的数据类型: {batch_labels.dtype}")

# 3. 随意抽查第一车里的第 0 张图，看看它是不是真的被处理成了 0 和 1
print("=== 抽查第一张图的最大最小值 ===")
print(f"这张图的最大像素值: {batch_images[0].max()}")
print(f"这张图的最小像素值: {batch_images[0].min()}")
# Model Definition