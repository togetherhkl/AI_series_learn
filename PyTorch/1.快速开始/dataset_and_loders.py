import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

#1 加载数据集
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()#将数据转换为张量
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

#2 迭代和可视化数据集
lables_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))#创建一个matplotlib图形对象，设置图形的大小为8x8英寸。
cols, rows = 3, 3#设置列数和行数
for i in range(1, cols * rows +1):#循环9次
    sample_idx = torch.randint(len(training_data),size=(1,)).item()
    '''
    使用torch.randint随机生成一个介于0和训练数据集长度之间的整数，作为随机选取的图像的索引。
    size=(1,)指定生成一个数，item()将其转换为Python的标准整数。
    '''
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)#添加子图,设置行数、列数和子图的索引,位置由i决定
    plt.title(lables_map[label])#设置标题
    plt.axis("off")#关闭坐标轴
    plt.imshow(img.squeeze(), cmap="gray")#灰度显示
    # plt.imshow(img.squeeze())#彩色显示，无需指定cmap
    '''
    img.squeeze()将图像张量的维度为1的轴删除，因为imshow函数预期的是一个二维图像。
    cmap="gray"指定了灰度图像。
    '''
plt.show()

#3 创建自定义数据集
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transfrom=None, target_tansform=None):
        self.img_labels = pd.read_csv(annotations_file)#读取CSV文件
        self.img_dir = img_dir#图像目录
        self.transfrom = transfrom#图像转换
        self.target_tansform = target_tansform#目标转换
    
    def __len__(self):
        return len(self.img_labels)#返回数据集的长度
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        imgage = read_image(img_path)#读取图像，转换成张量
        label = self.img_labels.iloc[idx, 1]#检索对应的标签
        if self.transfrom:#转换图像
            imgage = self.transfrom(imgage)
        if self.target_tansform:
            label = self.target_tansform(label)
        return imgage, label #以元组的形式返回图像和标签
    
#4 使用DataLoader准备训练数据
'''
在训练模型的过程中，通常希望以“小批量”的形式传递样本，
每个周期重新打乱数据以减少模型的过拟合，并使用Python的`multiprocessing`多进程来加速数据检索。
数据集（Dataset）负责逐个样本地获取数据集的特征和标签。
DataLoader是一个可迭代对象，它抽象了这些复杂性，提供了一个简单的API。
'''
from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
#在这里，DataLoader将训练数据集传递给train_dataloader，每个小批量包含64个特征和标签对,shuffle=True表示在每个周期重新打乱数据。
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

#5 通过DataLoader迭代
'''
将该数据集加载到DataLoader中，可以根据需要迭代遍历数据集。
每次迭代都会返回一批`train_features`和`train_labels`（分别包含`batch_size=64`个特征和标签）。
因为指定了`shuffle=True`，所以在遍历完所有的批次之后，数据会被重新打乱。
这意味着每个周期（epoch）开始时，数据的顺序都会随机化，有助于模型学习到更加泛化的特征，从而减少过拟合的风险。
'''
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
