# English version:

## Generally introduction:
  this project can be ued to recognize bikes' photos,which can uitilze CUDA to accelerate the project.
  After using data to train the module, this module can judge a photo whether it is a bike or not.Simultaneously,the data can be replace by ohter kind of thins because the data come from CLFAR,which includes various things(the detail can be seen in the data part below) .

## training data
the training data come from CLFAR,which includes CLFAR-10 and CLFAR-100. 
### CLFAR-10:
#### Core Dataset Information​Number of Classes
10 mutually exclusive object categories (6,000 images per class, totaling 60,000 images).
#### ​Image Dimensions
32×32-pixel RGB color images (extremely low resolution, optimized for fast training).
#### ​Data Split 
Training Set: 50,000 images (5,000 images per class). 
Test Set: 10,000 images (1,000 images per class).​
#### Data Source
Curated and labeled from the 80 Million Tiny Images dataset by Alex Krizhevsky.​2. The 10 Object Categories and Examples The CIFAR-10 dataset encompasses everyday objects, with the 10 categories explicitly defined as follows:
### CLFAR-100
#### Number of Classes
100 fine-grained object categories (e.g., subtypes of vehicles, plants, or medical conditions), with ​60,000–100,000 images (scaling proportionally to CIFAR-10’s 60,000 images).
#### ​Image Dimensions
High-resolution (e.g., 224×224 or 512×512 pixels), enabling detailed feature extraction for tasks like fine-grained classification and segmentation.
#### Data Split
Training Set: 80,000–90,000 images (800–900 images per class). Validation Set: 5,000 images (50 images per class). 
Test Set: 10,000–15,000 images (100–150 images per class).​Data Sources: Synthesized from ​real-world sensors (e.g., autonomous vehicles, medical imaging devices) and ​public datasets (e.g., ImageNet-21k, OpenImages). Annotated via ​semi-automated tools + ​human experts to ensure precision in fine-grained labels. 
#### addtion
in this project,I downloaded the data in the dirction named "cifar-100-python".If you don't want to download it in your computer,you can also ues it,it just has a difference in the code.
## coding part:

### utilizing CUDA
```python
import os 
# 检查GPU是否可用 to check if CUDA is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
```
on my device,I uesd CUDA 0.the uesr can change the number of the CUDA

### traning data collection
if you download the data in your computer and place it in the same dirction with the code.
```python
# 加载CIFAR-100数据集
train_set = torchvision.datasets.CIFAR100(root='./', train=True, download=False, transform=transform)
test_set = torchvision.datasets.CIFAR100(root='./', train=False, download=False, transform=transform)
```
'./' means use the data in the same dirction and 'download = False' means I have downloaded the data in my computer

### creating a label for bike
```python
#查找bicycle的索引 find the index of bicycle
bike_class_idx = train_set.classes.index('bicycle')
print(f"自行车（bicycle）的索引是: {bike_class_idx}")
print("查找成功")
# 创建自行车/非自行车二分类标签
def create_binary_labels(dataset,bike_class_idx):
    for i in range(len(dataset)):
        _, label = dataset[i]
        dataset.targets[i] = 1 if label == bike_class_idx else 0
```
if you ues this project for recognizing other things,you should find the right index and modify this part of codes.

## function checking
the project also inludes drawing a gragh to show the function of the module
```python
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
```
a traning_plot.png will be saved in the dirction.

## function test
```python
# 7. 测试单张图像
def predict_bike(image_tensor):
    """预测单张图像是否为自行车"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # 添加批次维度
        output = model(image_tensor)
        probability = output.item()
        return "bicycle" if probability > 0.5 else "not bicycle", probability
```
this part of code selets a image at random from the traning module to let the module judge whether it is a bicycle or not

## summarize
 this project still has a lot of weakness,weclome everyone to give suggestions or pull request.Any useful  issues will be carefully considered.If this project can help you,please give a star.