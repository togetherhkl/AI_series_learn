# 导入必要的库
import torch
from torch import nn# 神经网络模块
from torch.utils.data import DataLoader# 数据加载器
from torchvision import datasets# 数据集
from torchvision.transforms import ToTensor# 图像转换为张量

# 下载训练数据集
train_data = datasets.FashionMNIST(
    root="data",  # 数据存储的路径
    train=True,   # 指定下载的是训练数据集
    download=True,  # 如果数据不存在，则通过网络下载
    transform=ToTensor()  # 将图片转换为Tensor
)

# 下载测试数据集
test_data = datasets.FashionMNIST(
    root="data",  # 数据存储的路径
    train=False,  # 指定下载的是测试数据集
    download=True,  # 如果数据不存在，则通过网络下载
    transform=ToTensor()  # 将图片转换为Tensor
)

batch_size = 64# 批大小

# 创建数据加载器
train_dataloader = DataLoader(train_data, batch_size=batch_size)
#将dataset作为参数传入DataLoader，DataLoader会自动将数据分批，打乱数据，将数据加载到内存中
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for x,y in test_dataloader:
    print(f"Shape of x [N, C, H, W]: {x.shape}")
    #x.shape是一个4维张量，第一个维度是批大小，第二个维度是通道数，第三和第四维度是图像的高度和宽度
    print(f"Shape of y: {y.shape}, {y.dtype}")
    '''
    Shape of x [N, C, H, W]: torch.Size([64, 1, 28, 28])
    Shape of y: torch.Size([64]), torch.int64
    '''
    break

#创建模型
'''
在PyTorch中定义神经网络，创建一个继承自nn.Module的类。
在__init__函数中定义神经网络的层
在forward函数中定义数据在神经网络中的传播路径
为了加速神经网络的训练，可以使用GPU或者MPS来训练模型
'''
#使用cpu，gpu，mps的设备来训练模型
device =(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
#Using cuda device


class NeuralNetwork(nn.Module):
    def __init__(self):#定义神经网络的层
        super().__init__()#调用父类的构造函数
        self.flatten = nn.Flatten()#将28*28的图像展平为784的向量
        self.linear_relu_stack = nn.Sequential(#定义一个包含三个线性层的神经网络
            nn.Linear(28*28,512),#输入层
            nn.ReLU(),#激活函数
            nn.Linear(512,512),#隐藏层
            nn.ReLU(),#激活函数
            nn.Linear(512,10),#输出层
        )
    def forward(self,x):#定义数据在神经网络中的传播路径
        x = self.flatten(x)#将图像展平
        logits = self.linear_relu_stack(x)#将展平后的图像传入神经网络
        return logits#返回输出

model = NeuralNetwork().to(device)#将模型加载到设备上
print(model)
'''
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
'''

#优化模型参数
'''
在训练模型之前，需要定义损失函数（loss function）和优化器(optimizer)。
'''
loss_fn = nn.CrossEntropyLoss()#使用交叉熵损失函数
#使用随机梯度下降优化器,model.parameters()返回模型的参数，lr=1e-3是学习率
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

'''
在单个训练循环中，模型会对分批提供它的「训练数据集」进行「预测」
并通过「反向传播算法」预测误差以调整模型的参数。
'''
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)#数据集的大小
    model.train()#将模型设置为训练模式
    for batch, (X, y) in enumerate(dataloader):#遍历数据集
        X, y = X.to(device), y.to(device)#将数据加载到设备上

        # 计算预测误差
        pred = model(X)#对输入的数据进行预测
        loss = loss_fn(pred, y)#计算损失，差异越小，模型预测的越准确

        # 反向传播
        loss.backward()#反向传播算法
        optimizer.step()#优化器更新模型参数
        optimizer.zero_grad()#梯度清零

        if batch % 100 == 0:#每100个批次打印一次
            loss, current = loss.item(), (batch+1) * len(X)#打印损失和当前的批次的数据量
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            '''
            {loss:>7f}：表示损失值以浮点数形式打印，总宽度为7位，右对齐。
            {current:>5d}：表示当前处理的总数据量以整数形式打印，总宽度为5位，右对齐。
            {size:>5d}：表示整个数据集的大小以整数形式打印，总宽度为5位，右对齐
            '''
#检查模型在测试数据集上的性能，以确保它在学习
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)#数据集的大小
    num_batches = len(dataloader)#批次的数量
    model.eval()#将模型设置为评估模式
    test_loss, correct = 0, 0#初始化损失和正确的数量
    with torch.no_grad():#关闭梯度计算
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)#将数据加载到设备上
            pred = model(X)#对输入的数据进行预测
            test_loss += loss_fn(pred, y).item()#计算损失
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()#计算正确的数量
            '''
            pred.argmax(1)找出每个预测中概率最高的类别的索引，== y判断这些索引是否与真实标签相等。
            结果是一个布尔Tensor，通过.type(torch.float)转换为浮点数Tensor，
            然后使用.sum().item()计算并累加正确预测的总数。
            '''
        test_loss /= num_batches#计算平均损失
        correct /= size#计算正确率
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

'''
训练过程在多个迭代（周期）中进行。在每个周期中，模型学习参数以做出更好的预测。
在每个周期打印模型的准确率和损失；希望看到准确率随着每个周期的增加而提高，损失随着每个周期的减少
'''
epochs = 5#迭代次数
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)#训练模型
    test(test_dataloader, model, loss_fn)#测试模型
print("Done!")#训练完成
'''
运行结果
Epoch 1
-------------------------------
loss: 2.304268  [   64/60000]
loss: 2.284021  [ 6464/60000]
loss: 2.263621  [12864/60000]
loss: 2.259448  [19264/60000]
loss: 2.231920  [25664/60000]
loss: 2.221592  [32064/60000]
loss: 2.215944  [38464/60000]
loss: 2.191191  [44864/60000]
loss: 2.177027  [51264/60000]
loss: 2.141848  [57664/60000]
Test Error: 
 Accuracy: 58.7%, Avg loss: 2.137664

Epoch 2
-------------------------------
loss: 2.147467  [   64/60000]
loss: 2.139907  [ 6464/60000]
loss: 2.077062  [12864/60000]
loss: 2.094236  [19264/60000]
loss: 2.030329  [25664/60000]
loss: 1.982215  [32064/60000]
loss: 1.997371  [38464/60000]
loss: 1.923110  [44864/60000]
loss: 1.913458  [51264/60000]
loss: 1.835431  [57664/60000]
Test Error: 
 Accuracy: 61.3%, Avg loss: 1.839774
'''

#保存模型
'''
保存模型的一种常见方法是序列化内部状态字典（包含模型参数）
'''
torch.save(model.state_dict(), "./model.pth")
print("Saved PyTorch Model State to ./model.pth")
'''
Saved PyTorch Model State to ./model.pth
'''

#加载模型
'''
加载模型的过程包括重新创建模型结构，并将状态字典加载到其中
'''
model = NeuralNetwork().to(device)#创建模型,to(device)将模型加载到设备上
model.load_state_dict(torch.load("./model.pth"))#加载模型

#利用模型进行预测
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
model.eval()#将模型设置为评估模式
x, y = test_data[0][0], test_data[0][1]#获取测试数据
with torch.no_grad():#关闭梯度计算
    pred = model(x.to(device))#对输入的数据进行预测
    predicted, actual = classes[pred[0].argmax(0)], classes[y]#获取预测的类别和真实的类别
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
    '''
    Predicted: "Ankle boot", Actual: "Ankle boot"
    '''
