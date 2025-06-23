import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from torchvision.datasets import CIFAR100
import os
# 检查GPU是否可用 to check if CUDA is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
# 1. 数据加载与预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-100数据集
train_set = torchvision.datasets.CIFAR100(root='./', train=True, download=False, transform=transform)
test_set = torchvision.datasets.CIFAR100(root='./', train=False, download=False, transform=transform)
#查找bicycle的索引
bike_class_idx = train_set.classes.index('bicycle')
print(f"自行车（bicycle）的索引是: {bike_class_idx}")
print("查找成功")
# 创建自行车/非自行车二分类标签
def create_binary_labels(dataset,bike_class_idx):
    for i in range(len(dataset)):
        _, label = dataset[i]
        dataset.targets[i] = 1 if label == bike_class_idx else 0

# 应用二分类转换
create_binary_labels(train_set,bike_class_idx)
create_binary_labels(test_set,bike_class_idx)

# 创建数据加载器 - 在Windows上设置num_workers=0
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

# 2. 定义CNN模型
class CarClassifier(nn.Module):
    def __init__(self):
        super(CarClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            # 输入: 3x32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: 32x16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: 64x8x8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: 128x4x4
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 初始化模型
model = CarClassifier().to(device)

# 3. 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
def train_model():
    num_epochs = 10
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.float().to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计信息
            running_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 计算训练准确率
        train_accuracy = 100 * correct / total
        train_losses.append(running_loss / len(train_loader))
        train_acc.append(train_accuracy)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.float().to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                
                val_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss / len(test_loader))
        val_acc.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracy:.2f}%')
    
    return train_losses, val_losses, train_acc, val_acc

# 5. 绘制训练曲线
def plot_training(train_losses, val_losses, train_acc, val_acc):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='training losses')
    plt.plot(val_losses, label='val losses')
    plt.title('training and val losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='training acc')
    plt.plot(val_acc, label='val acc')
    plt.title('training and val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig('training_plot.png')

# 6. 保存模型
def save_model():
    torch.save(model.state_dict(), 'bike_classifier.pth')
    print("模型已保存为 'bike_classifier.pth'")

# 7. 测试单张图像
def predict_bike(image_tensor):
    """预测单张图像是否为自行车"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # 添加批次维度
        output = model(image_tensor)
        probability = output.item()
        return "bicycle" if probability > 0.5 else "not bicycle", probability

# 主函数
if __name__ == '__main__':
    # 修复Windows多进程问题
    multiprocessing.freeze_support()
    
    # 训练模型
    train_losses, val_losses, train_acc, val_acc = train_model()
    
    # 绘制训练曲线
    plot_training(train_losses, val_losses, train_acc, val_acc)
    plt.clf() 
    # 保存模型
    save_model()
    
    # 从测试集中随机选择一张图像
    sample_idx = np.random.randint(0, len(test_set))
    sample_img, sample_label = test_set[sample_idx]
    print(sample_idx)
    # 预测并显示结果
    prediction, prob = predict_bike(sample_img)
    true_label = "bicycle" if sample_label == 1 else "not bicycle"
        
    # 反归一化显示图像
    img = sample_img / 2 + 0.5  # 反归一化
    img_np = img.permute(1, 2, 0).numpy()  # 将PyTorch张量转换为numpy格式
    
    plt.imshow(img_np)
    plt.title(f"true: {true_label}\nprediction: {prediction} ({prob:.2f})")
    plt.axis('off')
#   plt.show()
    plt.savefig('output_image.jpg')
