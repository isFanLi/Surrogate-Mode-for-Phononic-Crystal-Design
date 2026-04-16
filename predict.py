import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class PhononicCNN(nn.Module): 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,5,padding=2), nn.ReLU(), nn.AvgPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,5,padding=2), nn.ReLU(), nn.AvgPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(32,64,5,padding=2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(64,128,5,padding=2), nn.ReLU(), nn.AvgPool2d(2))
        self.conv5 = nn.Sequential(nn.Conv2d(128,128,5,padding=2), nn.ReLU(), nn.AvgPool2d(2))
        self.conv6 = nn.Sequential(nn.Conv2d(128,256,5,padding=2), nn.ReLU(), nn.AvgPool2d(2))
        self.conv7 = nn.Sequential(nn.Conv2d(256,256,5,padding=2), nn.ReLU(), nn.AvgPool2d(2))

        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4, 2048), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(2048, 6656), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(6656, 1944), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1944, 1464)
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.dense_layers(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PhononicCNN().to(device)

model.load_state_dict(torch.load("best.pth", weights_only=True))
model.eval() 
print("best.pth 加载")


# 加载验证集
val_images = np.load(r'processed_phononic_9000\val_images.npy')
val_labels = np.load(r'processed_phononic_9000\val_labels.npy')

IMAGE_INDEX = 20

test_img_numpy = val_images[IMAGE_INDEX]  # 真实图片 (1, 256, 256)
ground_truth = val_labels[IMAGE_INDEX]    # 真实标签 (1464,)

# 把单张图片变成模型爱吃的格式：强行加一个 Batch 维度变成 [1, 1, 256, 256]
test_img_tensor = torch.tensor(test_img_numpy, dtype=torch.float32).unsqueeze(0).to(device)

# ==========================================
# 4. 让模型瞬间预测！
# ==========================================
with torch.no_grad(): # 考试结界，不记梯度
    prediction = model(test_img_tensor)

# 把预测结果从 GPU 拿回内存，并去掉那个假 Batch 维度，变成一维数组 (1464,)
pred_numpy = prediction.cpu().numpy().squeeze()

# ==========================================
# 5. 见证奇迹：画出物理色散图
# ==========================================
# 论文设定：1464 个点 = 24 条能带 (Bands) × 61 个波矢 K 点
NUM_BANDS = 24
NUM_K_POINTS = 61

# 把一根面条切成 24 根：(1464,) -> (24, 61)
pred_bands = pred_numpy.reshape((NUM_BANDS, NUM_K_POINTS))
truth_bands = ground_truth.reshape((NUM_BANDS, NUM_K_POINTS))

k_path = np.arange(NUM_K_POINTS) # X轴坐标

# 开始用 Matplotlib 绘图
plt.figure(figsize=(8, 6))
plt.title(f"Phononic Band Structure (Validation Image #{IMAGE_INDEX})", fontsize=14, fontweight='bold')

# 逐条画出 24 条线
# 逐条画出 24 条能带的点阵
for i in range(NUM_BANDS):
    if i == 0:
        # 真实值：红色实心圆点 (marker='o', linestyle='none' 取消连线)
        plt.plot(k_path, truth_bands[i, :], color='red', marker='o', linestyle='none', markersize=4, label='Ground Truth (FEM)')
        
        # 预测值：蓝色空心圆圈 (markerfacecolor='none' 实现空心效果)
        plt.plot(k_path, pred_bands[i, :], color='blue', marker='o', markerfacecolor='none', linestyle='none', markersize=6, label='CNN Prediction')
    else:
        # 后面的线不加 label，防止图例刷屏
        plt.plot(k_path, truth_bands[i, :], color='red', marker='o', linestyle='none', markersize=4)
        plt.plot(k_path, pred_bands[i, :], color='blue', marker='o', markerfacecolor='none', linestyle='none', markersize=6)

# 美化图表
plt.xlabel("Wave Vector (K-path index)", fontsize=12)
plt.ylabel("Normalized Frequency", fontsize=12)
plt.xlim(0, 60)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)

plt.show()