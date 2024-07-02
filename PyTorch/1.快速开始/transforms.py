import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))    
)
'''
    target_transform参数用于对目标（标签）进行转换。
    lamdba:定义一个匿名函数，y是函数的参数
    torch.zeros(10, dtype=torch.float)创建一个长度为10，数据类型为float的零向量
    .scatter_(0, torch.tensor(y), value=1)就地操作，在零向量的y索引的位置上的值设置为1，设置一个one-hot编码
    '''
add = lambda x, y:x+y
print(add(2,3))

fruits = [('apple', 2), ('danana', 3), ('cherry', 4)]
fruits.sort(key=lambda x:x[0])
print(fruits)