import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#1.获取训练的设备
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
# Using cuda device

#2.定义神经网络
'''
通过继承nn.Module来定义神经网络，并在__init__方法中初始化神经网络层。
每一个nn.Module的子类都在forward方法中实现了对输入数据的操作。
'''
class NueralNetwork(nn.Module):
    def __init__(self):#初始化神经网络层
        super().__init__()#调用父类的初始化方法
        self.flatten = nn.Flatten()#将图像张量展平
        self.linear_relu_stack = nn.Sequential(#定义一个包含三个全连接层的神经网络
            nn.Linear(28*28, 512),#输入层，参数分别为输入特征的形状和输出特征的形状
            nn.ReLU(),#激活函数
            nn.Linear(512, 512),#隐藏层，参数分别为输入特征的形状和输出特征的形状
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)#展平图像张量
        logits = self.linear_relu_stack(x)#将张量传递给神经网络
        return logits#返回输出
'''
实例化神经网络并将其移动到设备上，并打印模型的结构。
'''
model = NueralNetwork().to(device)
print(model)
'''
NueralNetwork(
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

'''
要使用模型，将「输入数据」传递给它。
这将执行模型的forward方法以及一些后台操作。但不要直接调用model.forward()。

调用模型对输入数据执行操作，将返回一个二维张量，
其中维度0对应于每个类别的10个原始预测值，维度1对应于每个输出的单独值。
我们通过传递给nn.Softmax模块的一个实例来获得预测概率。
'''
X = torch.rand(1, 28, 28, device=device)#生成一个随机张量,参数分别为张量的形状和设备，形状为1*28*28的张量
logits = model(X)#将张量传递给神经网络
pred_probab = nn.Softmax(dim=1)(logits)#将预测值传递给Softmax函数，dim=1表示计算每行的softmax,(logits)是「对象自调用」
y_pred = pred_probab.argmax(1)#返回每行中最大值的索引，若参数为0，则返回每列中最大值的索引
print(f"Predicted class: {y_pred}")#打印预测的类别
'''
Predicted class: tensor([1], device='cuda:0')
'''

#3.模型层

input_image = torch.rand(3, 28, 28)#随机生成一个3*28*28的张量
print(input_image.size())#打印张量的形状
print(input_image.shape)#打印张量的形状
'''
torch.Size([3, 28, 28])
torch.Size([3, 28, 28])
'''
# nn.Flatten
'''
初始化Flatten层后，可以通过调用它来展平3D张量。将3*28*28图像转换成一个连续的784像素值的数组。
'''
flatten = nn.Flatten() #实例化Flatten层
flat_image = flatten(input_image)#将张量传递给Flatten层
print(flat_image.size())#打印张量的形状
'''
torch.Size([3, 784])
'''

# nn.Linear
'''
Linear层使用一种称为「权重」的内部张量，以及一种称为「偏置」的内部张量，
对输入张量进行「线性变换（ linear transformation）」。
'''
layer1 = nn.Linear(in_features=28*28,out_features=20)#in_features表示输入特征的形状，out_features表示输出特征的形状
hidden1 = layer1(flat_image)#将张量传递给Linear层
print(hidden1.size())#打印张量的形状
'''
torch.Size([3, 20])
'''

# nn.ReLU
'''
「非线性激活函数」对模型的「输人」和「输出」创建「复杂的映射」。
在线性变换后，引入非线性激活函数，帮助神经网络学习各种规律。
'''
print(f"Before ReLU:{hidden1}\n\n")#打印隐藏层的输出
hidden1 = nn.ReLU()(hidden1)#将隐藏层的输出传递给ReLU激活函数
print(f"After ReLU: {hidden1}")#打印隐藏层的输出
'''
Before ReLU:tensor([[-0.6295, -0.0362, -0.1422,  0.1866, -0.0955,  0.1350, -0.0350, -0.0746,
         -0.3552,  0.2612, -0.1565, -0.1210, -0.1081,  0.0425,  0.3023,  0.0560,
          0.2418, -0.0035,  0.9525,  0.1108],
        [-0.6520, -0.3238, -0.0208,  0.0317,  0.0194,  0.5342, -0.2582, -0.3136,
         -0.3851,  0.2427, -0.0782, -0.3597, -0.2151, -0.1793, -0.0808, -0.1593,
          0.4785, -0.0835,  0.9555, -0.1394],
        [-0.9776, -0.3067, -0.1160, -0.0596,  0.1393,  0.2737,  0.1556,  0.0434,
         -0.6965,  0.4378, -0.2360, -0.1565,  0.3842, -0.2784,  0.3218, -0.0107,
          0.5351, -0.2072,  0.8570, -0.1982]], grad_fn=<AddmmBackward0>)


After ReLU: tensor([[0.0000, 0.0000, 0.0000, 0.1866, 0.0000, 0.1350, 0.0000, 0.0000, 0.0000,
         0.2612, 0.0000, 0.0000, 0.0000, 0.0425, 0.3023, 0.0560, 0.2418, 0.0000,
         0.9525, 0.1108],
        [0.0000, 0.0000, 0.0000, 0.0317, 0.0194, 0.5342, 0.0000, 0.0000, 0.0000,
         0.2427, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4785, 0.0000,
         0.9555, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.1393, 0.2737, 0.1556, 0.0434, 0.0000,
         0.4378, 0.0000, 0.0000, 0.3842, 0.0000, 0.3218, 0.0000, 0.5351, 0.0000,
         0.8570, 0.0000]], grad_fn=<ReluBackward0>)
'''

# nn.Sequential
'''
nn.Sequential是一个有序的容器，数据按照在函数中传递给它的顺序通过所有的模块。
可以使用nn.Sequential容器快速组装网络，例如seq_modules。
'''
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)#随机生成一个3*28*28的张量
logits = seq_modules(input_image)#将张量传递给Sequential容器

# nn.Softmax
'''
神经网络的最后一个线性层返回「原始预测值」，这些值被称为「logits」，
[-infty, infty]中的原始值传递给nn.Softmax模块，将其转换为[0, 1]范围内的值。
dim参数表示沿着哪个轴计算softmax。
'''
softmax = nn.Softmax(dim=1)#实例化Softmax函数
pred_probab = softmax(logits)#将logits传递给Softmax函数

#4.模型参数
'''
「神经网络」内部的许多「层」是「参数化」的，在训练过程中能够优化相关「权重」和「偏置」。
继承nn.Module的类会自动「模型对象」内部的定义的参数，
可以使用模型的parameters()或named_parameters()方法使所有参数可访问。
'''
print("Model structure: ", model, "\n\n")#打印模型的结构
for name, param in model.named_parameters():#遍历模型的参数
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")#打印参数的名称、形状和前两个值
'''
Model structure:  NueralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values: tensor([[ 1.2033e-02, -3.3190e-02,  3.5117e-02,  ..., -3.1082e-04,
          3.1766e-02,  8.9217e-05],
        [-2.2151e-02,  8.2360e-03,  2.6249e-02,  ...,  1.1201e-02,
          1.0973e-02,  3.0528e-02]], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values: tensor([0.0314, 0.0233], device='cuda:0', grad_fn=<SliceBackward0>)       

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values: tensor([[-0.0299, -0.0194,  0.0357,  ..., -0.0063, -0.0406,  0.0399],
        [ 0.0007,  0.0034,  0.0072,  ...,  0.0176, -0.0431,  0.0424]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values: tensor([0.0287, 0.0206], device='cuda:0', grad_fn=<SliceBackward0>)       

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values: tensor([[ 0.0293,  0.0368, -0.0042,  ..., -0.0112, -0.0114, -0.0138],
        [ 0.0157,  0.0046, -0.0023,  ..., -0.0414, -0.0390, -0.0082]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values: tensor([0.0046, 0.0029], device='cuda:0', grad_fn=<SliceBackward0>) 
'''  