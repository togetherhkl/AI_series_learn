'''
现在我们已经拥有了一个「模型」和「数据」，是时候通过「优化模型参数」来训练、验证和测试我们的模型了。
训练模型是一个迭代过程；在每次迭代中，模型对输出做出猜测，计算其猜测的误差（损失），
收集误差相对于其参数的导数（正如我们在前一节看到的），并使用梯度下降来优化这些参数。
'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# 加载数据集
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# 创建数据加载器
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
# 创建模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# 超级参数
'''
超参数是可调节的参数，它们允许你控制模型优化过程。不同的超参数值可能会影响模型的训练和收敛速率（了解更多关于超参数调整的信息）。

我们为训练定义了以下超参数：

周期数（Epochs）：迭代数据集的次数。
批量大小（Batch Size）：在更新参数之前通过网络传播的数据样本数量。
学习率（Learning Rate）：在每个批量/周期中更新模型参数的程度。较小的值会导致学习速度慢，而较大的值可能会导致训练过程中出现不可预测的行为。
'''
learning_rate = 1e-3
batch_size = 64
epochs = 5

# 优化循环
'''
一旦设置了超参数，就可以通过优化循环来训练和优化模型。优化循环的每一次迭代称为一个周期（Epoch）。

每个周期由两个主要部分组成：

训练循环（Train Loop）：遍历训练数据集，尝试收敛到最优参数。
验证/测试循环（Validation/Test Loop）：遍历测试数据集，以检查模型性能是否在提高。

'''

# 损失函数
'''
当提供一些训练数据时，未经训练的网络很可能无法给出正确的答案。
损失函数衡量了所得结果与目标值之间的不相似程度，而在训练过程中，
希望最小化的就是这个损失函数。为了计算损失，使用给定数据样本的输入进行预测，并将其与真实的数据标签值进行比较。

常见的损失函数包括：

nn.MSELoss（均方误差）：用于回归任务。
nn.NLLLoss（负对数似然损失）：用于分类任务。
nn.CrossEntropyLoss：结合了nn.LogSoftmax和nn.NLLLoss的功能。
将模型的输出（logits）传递给nn.CrossEntropyLoss，它将对logits进行归一化并计算预测误差。
'''
loss_fn = nn.CrossEntropyLoss()#初始化损失函数

# 优化器
'''
优化是调整模型参数以减少每个训练步骤中的模型误差的过程。
优化算法定义了这个过程是如何执行的（在这个例子中，使用随机梯度下降）。
所有的优化逻辑都被封装在优化器`optimizer`对象中。在这里，使用SGD（随机梯度下降）优化器；
此外，在PyTorch中还有许多不同的优化器可用，例如ADAM和RMSProp，
它们对不同类型的模型和数据有更好的效果。

通过注册需要训练的模型参数，并传入学习率超参数来初始化优化器。
'''
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)#初始化优化器

'''
在训练循环中，优化发生在三个步骤中：

梯度清零：调用optimizer.zero_grad()来重置模型参数的梯度。梯度默认情况下是累加的；为了防止重复计算，在每次迭代中明确地将它们清零。

反向传播：通过调用loss.backward()对预测损失进行反向传播。PyTorch会计算损失相对于每个参数的梯度。

参数更新：一旦有了梯度，就调用optimizer.step()根据反向传播过程中收集的梯度来调整参数。

这个过程确保了模型在每次迭代中都能朝着减少损失的方向更新参数。
'''

# 完整训练循环
'''
定义循环优化代码的train_loop，以及根据测试数据评估模型性能的test_loop。
'''
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)#数据集的大小
    model.train()#设置模型为训练模式
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)#前向传播
        loss = loss_fn(pred, y)

        # 反向传播
        loss.backward()
        optimizer.step()#参数更新
        optimizer.zero_grad()#梯度清零

        if batch % 100 == 0:#每100个批次打印一次
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()#设置模型为评估模式
    size = len(dataloader.dataset)#数据集的大小
    test_loss, correct = 0, 0#测试损失和正确数
    num_batches = len(dataloader)#批次数

    with torch.no_grad():#关闭梯度跟踪
        for X, y in dataloader:
            pred = model(X)#前向传播
            test_loss += loss_fn(pred, y).item()#计算损失
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()#计算正确数
    test_loss /= num_batches#计算平均损失
    correct /= size#计算正确率
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

'''
初始化损失函数和优化器，并将其传递给train_loop和test_loop。随意增加轮数以跟踪模型的改进性能。
'''
loss_fn = nn.CrossEntropyLoss()#初始化损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)#初始化优化器
epochs = 10#周期数

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")