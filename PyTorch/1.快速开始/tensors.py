import torch
import numpy as np

# 初始化张量
#1.数据直接创建
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

#2.从numpy数组创建
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#3.从另一个张量创建
x_ones = torch.ones_like(x_data)#保留x_data的属性
print(f"Ones Tensor: \n {x_ones} \n")
'''
Ones Tensor: 
 tensor([[1, 1],
        [1, 1]]) 
'''

x_rand = torch.rand_like(x_data, dtype=torch.float)#覆盖数据类型
print(f"Random Tensor: \n {x_rand} \n")
'''
Random Tensor: 
 tensor([[0.3233, 0.8274],
        [0.6734, 0.7567]]) 
'''

#4.使用随机或常量值创建
shape = (2,3,)#2行3列
rand_tensor = torch.rand(shape)#随机张量
ones_tensor = torch.ones(shape)#全1张量
zeros_tensor = torch.zeros(shape)#全0张量
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
'''
Random Tensor:
 tensor([[0.8509, 0.5273, 0.0969],
        [0.0020, 0.7490, 0.3388]])

Ones Tensor:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
'''

#张量的属性
'''
张量的属性描述了它们的形状、数据类型以及它们存储的设备。
'''
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
'''
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
'''

#张量的操作

#1.移动到GPU
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

#2.标准的numpy索引和切片
tensor = torch.ones(4,4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)
'''
First row:  tensor([1., 1., 1., 1.])
First column:  tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
'''
#3.张量连接
t1 = torch.cat([tensor, tensor, tensor], dim=1)#按列连接,dim=0按行连接
print(t1)
'''
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
'''
#4.算术运算
#4.1 矩阵乘法
y1 = tensor @ tensor.T#矩阵乘法,tensor.T表示转置
y2 = tensor.matmul(tensor.T)#矩阵乘法,等价于@
y3 = torch.rand_like(tensor)#随机张量
torch.matmul(tensor, tensor.T, out=y3)#矩阵乘法,指定输出,out=y3表示将结果存储在y3中
print(y3)
'''
tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
'''
#4.2 元素级乘法
z1 = tensor * tensor#元素级乘法
z2 = torch.mul(tensor, tensor)#元素级乘法,等价于*
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)#元素级乘法,指定输出
print(z3)
'''
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
'''
#5.单元素张量
'''
如果有一个单元素张量，例如通过将张量的所有值聚合为一个值，可以使用 item() 方法将其转换为Python数值。
'''
agg = tensor.sum()#聚合
agg_item = agg.item()
print(agg_item, type(agg_item))
'''
12.0 <class 'float'>
'''
#6.就地操作
'''
就地操作 将结果存储到操作数中的操作称为就地操作。它们以 _ 后缀表示。例如：x.copy_(y), x.t_()，将会改变 x。
'''
print(tensor, "\n")
tensor.add_(5)#加5
print(tensor)

