import sys
import numpy as np
import cv2
import os
import dlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
from PIL import Image
import torch.nn.functional as F

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')#忽略警告
class simpleconv3(nn.Module):#定义一个简单的三层卷积神经网络
    def __init__(self):
        super(simpleconv3,self).__init__()#复制并使用simpleconv3的父类的初始化方法，即先运行nn.Module的初始化函数
        self.conv1 = nn.Conv2d(3, 12, 3, 2)#定义conv1函数的是图像卷积层：输入是3个feature，输出是6个feature，kernel是3*3，步长为2
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, 3, 2)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 48, 3, 2)
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48 * 5 * 5 , 1200)
        self.fc2 = nn.Linear(1200 , 128)
        self.fc3 = nn.Linear(128 , 4)

    def forward(self , x):
        x = F.relu(self.bn1(self.conv1(x)))
        #print "bn1 shape",x.shape
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1 , 48 * 5 * 5) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

PREDICTOR_PATH = "./Emotion_Recognition_File/face_detect_model/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)#加载人脸关键点检测模型
cascade_path = './Emotion_Recognition_File/face_detect_model/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)#加载人脸检测器

if not os.path.exists("results"):
    os.mkdir("results")
    

def standardization(data):
    '''
    功能：标准化数据
    参数：data：数据
    返回：标准化后的数据
    '''
    mu = np.mean(data, axis=0)#计算每一列的均值，axis=0表示列，axis=1表示行
    sigma = np.std(data, axis=0)#计算每一列的标准差，axis表示方向
    return (data - mu) / sigma
    '''
    数据按列减去其平均值后，再除以其标准差。
    这一步是标准化的核心，目的是让处理后的数据每一列的平均值为0，标准差为1。
    这样处理后的数据符合标准正态分布，有利于后续的数据处理和分析。
    '''


def get_landmarks(im):
    '''
    功能：获取人脸关键点
    参数：im：图像
    返回：人脸关键点
    '''
    rects = cascade.detectMultiScale(im, 1.3, 5)#检测人脸
    x, y, w, h = rects[0]#获取人脸区域
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))#转换为dlib格式的人脸区域
    return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])#获取人脸关键点


def annotate_landmarks(im, landmarks):
    '''
    功能：标注人脸关键点
    参数：im：图像
          landmarks：人脸关键点
    返回：标注后的图像
    '''
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im,
                    str(idx),
                    pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


testsize = 48  # 测试图大小

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
net = simpleconv3()#创建模型实例
net.eval()#评估模式
modelpath = "./models/model.ckpt"  # 模型路径
net.load_state_dict(#加载模型保存的模型权重
    torch.load(modelpath, map_location=lambda storage, loc: storage))
'''
    map_location参数用于指定加载模型的设备位置。
    这里使用了一个lambda函数lambda storage, loc: storage，
    这意味着无论模型权重文件是在哪个设备上保存的（例如GPU或CPU），
    都将其加载到当前设备上。这对于跨设备加载模型非常有用，特别是当原始训练环境与当前环境不同时。
    lambda语法：lambda 参数:表达式
    '''

# 一次测试一个文件
img_path = "./Emotion_Recognition_File/test_img/"
imagepaths = os.listdir(img_path)  # 图像文件夹
for imagepath in imagepaths:
    im = cv2.imread(os.path.join(img_path, imagepath), 1)
    try:
        rects = cascade.detectMultiScale(im, 1.3, 5)#检测人脸
        x, y, w, h = rects[0]
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        landmarks = np.matrix([[p.x, p.y]
                               for p in predictor(im, rect).parts()])
    except:
#         print("没有检测到人脸")
        continue  # 没有检测到人脸

    xmin = 10000
    xmax = 0
    ymin = 10000
    ymax = 0

    for i in range(48, 67):
        x = landmarks[i, 0]
        y = landmarks[i, 1]
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y

    roiwidth = xmax - xmin
    roiheight = ymax - ymin

    roi = im[ymin:ymax, xmin:xmax, 0:3]

    if roiwidth > roiheight:
        dstlen = 1.5 * roiwidth
    else:
        dstlen = 1.5 * roiheight

    diff_xlen = dstlen - roiwidth
    diff_ylen = dstlen - roiheight

    newx = xmin
    newy = ymin

    imagerows, imagecols, channel = im.shape
    if newx >= diff_xlen / 2 and newx + roiwidth + diff_xlen / 2 < imagecols:
        newx = newx - diff_xlen / 2
    elif newx < diff_xlen / 2:
        newx = 0
    else:
        newx = imagecols - dstlen

    if newy >= diff_ylen / 2 and newy + roiheight + diff_ylen / 2 < imagerows:
        newy = newy - diff_ylen / 2
    elif newy < diff_ylen / 2:
        newy = 0
    else:
        newy = imagecols - dstlen

    roi = im[int(newy):int(newy + dstlen), int(newx):int(newx + dstlen), 0:3]#裁剪图像
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)#转换颜色空间将BGR转为RGB
    roiresized = cv2.resize(roi,#调整图像大小
                            (testsize, testsize)).astype(np.float32) / 255.0
    imgblob = data_transforms(roiresized).unsqueeze(0)#转换数据格式
    imgblob.requires_grad = False#不需要梯度，因为是验证阶段而不是训练阶段
    imgblob = Variable(imgblob)#转换数据格式
    torch.no_grad()#通过上下文管理器禁用梯度计算
    predict = F.softmax(net(imgblob))#net(imgblob)是模型的输出，F.softmax()是softmax函数，用于多分类问题
    print(predict)
    index = np.argmax(predict.detach().numpy())#获取预测结果，np.argmax()返回概率最大的类别

    im_show = cv2.imread(os.path.join(img_path, imagepath), 1)
    im_h, im_w, im_c = im_show.shape#获取图像的高、宽、通道数
    pos_x = int(newx + dstlen)
    pos_y = int(newy + dstlen)
    font = cv2.FONT_HERSHEY_SIMPLEX#设置字体样式
    cv2.rectangle(im_show, (int(newx), int(newy)),#画矩形框
                  (int(newx + dstlen), int(newy + dstlen)), (0, 255, 255), 2)
    if index == 0:
        cv2.putText(im_show, 'none', (pos_x, pos_y), font, 1.5, (0, 0, 255), 2)
    if index == 1:
        cv2.putText(im_show, 'pout', (pos_x, pos_y), font, 1.5, (0, 0, 255), 2)
    if index == 2:
        cv2.putText(im_show, 'smile', (pos_x, pos_y), font, 1.5, (0, 0, 255), 2)
    if index == 3:
        cv2.putText(im_show, 'open', (pos_x, pos_y), font, 1.5, (0, 0, 255), 2)
#     cv2.namedWindow('result', 0)
#     cv2.imshow('result', im_show)
    cv2.imwrite(os.path.join('results', imagepath), im_show)
#     print(os.path.join('results', imagepath))
    plt.imshow(im_show[:, :, ::-1])  # 这里需要交换通道，因为 matplotlib 保存图片的通道顺序是 RGB，而在 OpenCV 中是 BGR
    plt.show()
#     cv2.waitKey(0)
# cv2.destroyAllWindows()