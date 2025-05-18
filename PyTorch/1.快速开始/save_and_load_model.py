# 保存与加载模型权重
import torch
import torchvision.models as models
'''
PyTorch保存模型的「学习参数」是通过`state_dict`的一个内部状态字典，使用`torch.save`来保存模型的学习参数。
'''

model = models.vgg16(weights='IMAGENET1K_V1')
'''
vgg16是一个非常流行的卷积神经网络，经过了大量的训练，可以识别1000个不同的对象。
weights='IMAGENET1K_V1'表示加载了在ImageNet数据集上预训练的权重。
'''
torch.save(model.state_dict(), 'model_weights.pth')#状态字典与保存路径

'''
加载模型权重，首先需要创建一个与原始模型相同的模型实例，然后使用`load_state_dict`方法加载参数。
注意：需要使用`model.eval()`方法将模型设置为评估模式，这将关闭Dropout和BatchNorm层。
否则将会导致不一致的推理结果。
'''
model = models.vgg16()#加载模型
model.load_state_dict(torch.load('model_weights.pth'))#加载模型权重
model.eval()#设置模型为评估模式

#保存和加载模型架构
'''
在加载模型权重时，需要首先实例化模型类，因为模型类定义了网络的结构。
如果希望将模型类的架构与模型一起保存，
那么可以传递模型本身（而不是模型的状态字典model.state_dict()）给保存函数。
'''
torch.save(model, 'model.pth')#保存模型
model = torch.load('model.pth')#加载模型

