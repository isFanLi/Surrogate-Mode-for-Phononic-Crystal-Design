import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import numpy as np

# Data Preparation
# 1.把.npy读到内存里，变成numpy数组
image_train_numpy = np.load(r'processed_phononic_9000\train_images.npy')
label_train_numpy = np.load(r'processed_phononic_9000\train_labels.npy')
# 2.把numpy数组变成变成tensor
image_train_tensor = torch.tensor(image_train_numpy, dtype=torch.float32)
label_train_tensor = torch.tensor(label_train_numpy, dtype=torch.float32)
# 3.把两个tensor合成一个dataset(Dict字典，前索引，后label)
train_set = TensorDataset(image_train_tensor, label_train_tensor)
# 4.把dataloader负责把dataset分成一个个小batch，分批加载进trinloader
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# 验证测试(筛选最佳参数文件)
image_val_numpy = np.load(r'processed_phononic_9000\val_images.npy')
label_val_numpy = np.load(r'processed_phononic_9000\val_labels.npy')

image_val_tensor = torch.tensor(image_val_numpy, dtype=torch.float32)
label_val_tensor = torch.tensor(label_val_numpy, dtype=torch.float32)

val_set = TensorDataset(image_val_tensor, label_val_tensor)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

# Model Definition
class PhononicCNN(nn.Module): # 继承nn.Module
    def __init__(self):
        super().__init__() # 父类初始化方法
        # Convolution Layers
        # 256 256
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,padding=2),nn.ReLU(),nn.AvgPool2d(2))
        # 128 128
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,padding=2),nn.ReLU(),nn.AvgPool2d(2))
        # 64 64 
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2),nn.ReLU(),nn.MaxPool2d(2))       # 注意第三层用最大池化
        # 32 32
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,padding=2),nn.ReLU(),nn.AvgPool2d(2))
        # 16 16
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=128,out_channels=128,kernel_size=5,padding=2),nn.ReLU(),nn.AvgPool2d(2))
        # 8 8 
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=128,out_channels=256,kernel_size=5,padding=2),nn.ReLU(),nn.AvgPool2d(2))
        # 4 4
        self.conv7 = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=256,kernel_size=5,padding=2),nn.ReLU(),nn.AvgPool2d(2))
        # 2 2

        # Dense Layers
        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            # Linear 1
            nn.Linear(in_features=256*4,out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            # Linear 2
            nn.Linear(in_features=2048,out_features=6656),
            nn.ReLU(),
            nn.Dropout(p=0.2),
             # Linear 3
            nn.Linear(in_features=6656,out_features=1944),
            nn.ReLU(),
            nn.Dropout(p=0.2),
             # Linear 4
            nn.Linear(in_features=1944,out_features=1464),
            # nn.ReLU(),
            # nn.Dropout(p=0.2) 
            # 为什么最后一层不能RELU和Dropout？
            # relu是把负数变0，但是反向传播的时候0不能求导?
            # dropout是把神经元的值变成0，最后一层加了就出问题
        )

     # forward
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
        
# test
# model = PhononicCNN()
# test_input = torch.randn(32,1,256,256)
# test_output = model(test_input)
# print(f"输入数据的维度: {test_input.shape}")
# print(f"模型吐出的维度: {test_output.shape}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = PhononicCNN()
model = model.to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(),lr=1.433e-4)

epoches = 100

# 初始误差设为无限大
best_val_loss = float('inf')

for epoch in range(epoches):
    # 训练阶段
    model.train()

    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_train_loss = running_loss / len(train_loader)
    # 验证阶段
    model.eval()

    val_running_loss = 0.0

    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_labels)
            val_running_loss += val_loss.item()
        # 平均误差
        epoch_val_loss = val_running_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epoches}] | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "best.pth")
        print("更新最佳模型")
    
torch.save(model.state_dict(), "last.pth")
print("训练结束")


        



