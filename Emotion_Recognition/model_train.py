from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import numpy as np

import warnings

# 定义一个简单的卷积神经网络
class simpleconv3(nn.Module):
    def __init__(self):
        super(simpleconv3,self).__init__()#调用父类的构造函数
        self.conv1 = nn.Conv2d(3, 12, 3, 2)#定义第一个卷积层
        self.bn1 = nn.BatchNorm2d(12)#定义第一个批标准化层
        self.conv2 = nn.Conv2d(12, 24, 3, 2)#定义第二个卷积层
        self.bn2 = nn.BatchNorm2d(24)#定义第二个批标准化层
        self.conv3 = nn.Conv2d(24, 48, 3, 2)#定义第三个卷积层
        self.bn3 = nn.BatchNorm2d(48)#定义第三个批标准化层
        self.fc1 = nn.Linear(48 * 5 * 5 , 1200)#定义第一个全连接层
        self.fc2 = nn.Linear(1200 , 128)#定义第二个全连接层
        self.fc3 = nn.Linear(128 , 4)#定义第三个全连接层

    def forward(self , x):#前向传播
        x = F.relu(self.bn1(self.conv1(x)))#卷积层->批标准化层->激活函数
        #print "bn1 shape",x.shape
        x = F.relu(self.bn2(self.conv2(x)))#卷积层->批标准化层->激活函数
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1 , 48 * 5 * 5) #reshape操作
        x = F.relu(self.fc1(x))#全连接层->激活函数
        x = F.relu(self.fc2(x))
        x = self.fc3(x)#全连接层
        return x

warnings.filterwarnings('ignore')#忽略警告

writer = SummaryWriter()#创建一个SummaryWriter对象，用于记录训练过程中的损失和准确率

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    '''
    功能：训练神经网络
    参数：  model，表示要训练的神经网络
            criterion，表示损失函数
            optimizer，表示优化器
            scheduler，表示学习率调度器
            num_epochs，表示训练的轮次
    返回值：model，表示训练好的神经网络
    '''
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))# 打印训练次数
        for phase in ['train', 'val']:#训练和验证
            if phase == 'train':#训练阶段
                scheduler.step()#学习率调度器更新学习率
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0#累计当前阶段的总损失，有助于计算平均损失
            running_corrects = 0.0#累计当前阶段的总准确率，有助于计算平均准确率

            for data in dataloders[phase]:#遍历数据加载器
                inputs, labels = data#获取输入数据和标签
                if use_gpu:#使用GPU
                    inputs = Variable(inputs.cuda())#将输入数据转换为Variable类型
                    labels = Variable(labels.cuda())#将标签转换为Variable类型
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()#进行一次前向传播之前，清零之前的梯度，以防止梯度累积
                outputs = model(inputs)#将输入数据喂给模型，进行一次前向传播，得到模型的输出。
                _, preds = torch.max(outputs.data, 1)#从模型输出中找到每个样本预测概率最高的类别作为预测结果。
                loss = criterion(outputs, labels)#计算模型输出和标签之间的损失
                if phase == 'train':#训练阶段
                    loss.backward()#进行一次反向传播，计算梯度
                    optimizer.step()#更新模型参数

                running_loss += loss.data.item()#累计当前阶段的总损失
                running_corrects += torch.sum(preds == labels).item()#累计当前阶段的总准确率

            epoch_loss = running_loss / dataset_sizes[phase]#计算平均损失
            epoch_acc = running_corrects / dataset_sizes[phase]#计算平均准确率
           
            if phase == 'train':
                writer.add_scalar('data/trainloss', epoch_loss, epoch)#记录训练阶段的平均损失
                writer.add_scalar('data/trainacc', epoch_acc, epoch)#记录训练阶段的平均准确率
            else:
                writer.add_scalar('data/valloss', epoch_loss, epoch)#记录验证阶段的平均损失
                writer.add_scalar('data/valacc', epoch_acc, epoch)#记录验证阶段的平均准确率

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

    writer.export_scalars_to_json("./all_scalars.json")#将记录的数据保存为json文件
    writer.close()#关闭SummaryWriter对象
    return model

if __name__ == '__main__':

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(48),# 随机裁剪，调整为48*48像素大小，增强泛化
            transforms.RandomHorizontalFlip(),# 随机水平翻转，增强泛化
            transforms.ToTensor(),# 转换为张量
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])# 标准化
        ]),
        'val': transforms.Compose([
            transforms.Resize(64),# 调整为64*64像素大小
            transforms.CenterCrop(48),# 中心裁剪，调整为48*48像素大小
            transforms.ToTensor(),# 转换为张量
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])# 标准化
        ]),
    }

    data_dir = './Emotion_Recognition_File/train_val_data/' # 数据集所在的位置
    #创建图像数据集
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'val']}
    #创建数据加载器
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],# 数据集
                                                 batch_size=64,# 每个批次加载64个样本
                                                 shuffle=True if x=="train" else False,# 训练集打乱，验证集不打乱
                                                 num_workers=8) for x in ['train', 'val']}# 使用8个子进程加载数据

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}# 数据集大小

    use_gpu = torch.cuda.is_available()# 是否使用GPU
    print("是否使用 GPU", use_gpu)
    modelclc = simpleconv3()# 创建模型
    print(modelclc)
    if use_gpu:
        modelclc = modelclc.cuda()# 使用GPU

    #配置神经网络训练过程中的损失函数、优化器和学习率调度器的。
    criterion = nn.CrossEntropyLoss()#选择交叉熵损失函数
    optimizer_ft = optim.SGD(modelclc.parameters(), lr=0.1, momentum=0.9)
    '''
    随机梯度下降（SGD）优化器，用于更新模型的权重。
    modelclc.parameters() 提供了模型中所有可训练的参数。
    lr=0.1 设置了学习率为 0.1，这是在训练过程中每次参数更新的步长。
    momentum=0.9 设置了动量为 0.9，这有助于加速优化器在相关方向上的速度，并减少震荡。
    '''
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
    '''
    这行代码定义了一个学习率调度器，它会在每过 step_size（这里是100个）训练轮次后，
    将学习率乘以 gamma（这里是0.1）。这意味着每100个训练轮次，学习率会减少到原来的10%。
    这种方法有助于模型在训练早期快速收敛，并在训练后期通过减小学习率来细化模型参数，以避免过拟合
    '''
    modelclc = train_model(model=modelclc,# 模型
                           criterion=criterion,# 损失函数
                           optimizer=optimizer_ft,# 优化器
                           scheduler=exp_lr_scheduler,# 学习率调度器
                           num_epochs=10)  # 这里可以调节训练的轮次
    #保存模型
    if not os.path.exists("models"):
        os.mkdir('models')

    torch.save(modelclc.state_dict(),'models/model.ckpt')# 保存模型参数