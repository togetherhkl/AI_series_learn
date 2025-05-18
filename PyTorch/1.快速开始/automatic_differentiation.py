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
默认情况，张量在tensor.requires_grad=True，都会跟踪它们的计算历史并支持梯度计算。
然而，在某些情况下，我们不需要这样做，例如，当我们已经训练了模型并且只想将其应用于某些输入数据时，
也就是说，我们只想通过网络执行前向计算。我们可以通过将计算代码包围在torch.no_grad()块中来停止跟踪计算
'''
z = torch.matmul(x, w)+b#矩阵乘法
print(z.requires_grad)#打印z是否需要计算梯度

with torch.no_grad():#禁用梯度跟踪
    z = torch.matmul(x, w)+b#矩阵乘法
print(z.requires_grad)#打印z是否需要计算梯度
'''
True
False
'''

#另一种方法是使用detach()方法从计算历史中分离张量
z = torch.matmul(x, w)+b#矩阵乘法
z_det = z.detach()#分离张量
print(z_det.requires_grad)#打印z_det是否需要计算梯度
'''
False
'''

'''
可能想要禁用梯度跟踪的原因有：

1. 将神经网络中的一些参数标记为冻结参数。
2. 当只进行前向传播时加速计算，因为不跟踪梯度的张量的计算将更加高效。
'''

#更多计算图
'''

从概念上讲，「自动微分（autograd）」保留了「数据（张量）」和所有执行的操作（以及由此产生的新张量）的记录，
在由Function对象组成的「有向无环图（DAG）」中。在这个DAG中，叶子是输入张量，根是输出张量。
通过从根到叶子追踪这个图，可以使用「链式法则」自动计算梯度。


在前向传播中，「自动微分（autograd）」同时做两件事：

1. 执行请求的操作来计算结果张量。
2. 在DAG中维护操作的「梯度函数」。

当在DAG根上调用`.backward()`时，开始反向传播。自动微分然后：

1. 从每个`.grad_fn`计算梯度，
2. 将它们累积在相应张量的`.grad`属性中，
3. 使用链式法则，一直传播到叶张量。


在PyTorch中，DAG（有向无环图）是动态的。
一个重要的特点是，每次调用`.backward()`之后，计算图都会从头开始重新创建。
这正是允许在模型中使用控制流语句的原因；如果需要，可以在每次迭代中改变形状、大小和操作。
'''

