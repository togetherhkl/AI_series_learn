'''
在训练神经网络时，最常用的算法是「反向传播 back propagation」。在这个算法中，参数（「模型权重」）根据「损失函数」的「梯度」进行调整。
为了计算这些梯度，PyTorch有一个名为torch.autograd的内置微分引擎。它支持任何计算图的梯度自动计算。
'''
import torch
x = torch.ones(5)#创建一个全1张量，形状为5*1
y = torch.zeros(3)#创建一个全0张量
w = torch.rand(5, 3, requires_grad=True)#创建一个随机张量,形状为5*3,需要计算梯度
b = torch.rand(3, requires_grad=True)#创建一个随机张量,形状为3*1,需要计算梯度
z = torch.matmul(x, w)+b#矩阵乘法
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)#损失函数为二进制交叉熵

print(f"Gradient function for z = {z.grad_fn}")#打印z的梯度函数
print(f"Gradient function for loss = {loss.grad_fn}")#打印loss的梯度函数
'''
Gradient function for z = <AddBackward0 object at 0x00000276285A12A0>
Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0x00000276285A12A0>
'''

#计算梯度
'''

为了优化神经网络中参数的权重，我们需要计算损失函数相对于参数的导数，即我们需要在固定的x和y值下计算
∂
loss
∂
𝑤
∂w
∂loss
​
 
和
∂
loss
∂
𝑏
∂b
∂loss
​
 
。为了计算这些导数，我们调用loss.backward()，然后从w.grad和b.grad中检索值
'''
loss.backward()#计算梯度
print(w.grad)#打印w的梯度
print(b.grad)#打印b的梯度
'''
tensor([[0.3175, 0.3126, 0.3231],
        [0.3175, 0.3126, 0.3231],
        [0.3175, 0.3126, 0.3231],
        [0.3175, 0.3126, 0.3231],
        [0.3175, 0.3126, 0.3231]])
tensor([0.3175, 0.3126, 0.3231])
'''

#禁用梯度跟踪
'''
默认情况下tensor.requires_grad_(True)张量都会跟踪它们的计算历史并支持梯度计算。
然而，在某些情况下，我们不需要这样做，例如，当我们已经训练了模型并且只想将其应用于某些输入数据时，
也就是说，我们只想通过网络执行前向计算。我们可以通过将计算代码包围在torch.no_grad()块中来停止跟踪计算
'''
z = torch.matmul(x, w)+b#矩阵乘法
print(z.requires_grad)#打印z是否需要计算梯度

with torch.no_grad():#禁用梯度跟踪
    z = torch.matmul(x, w)+b#矩阵乘法
print(z.requires_grad)#打印z是否需要计算梯度