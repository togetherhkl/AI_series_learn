'''
使用OpenCV检测图片中的人脸，并显示出来，删除没有人脸的图片
使用到的知识：
1. OpenCV 的人脸检测接口
2. os.listdir() 方法，用于返回指定的文件夹包含的文件或文件夹的名字的列表
3. os.remove() 方法，用于删除文件
4. plt.imshow() 方法，用于显示图片
5. plt.show() 方法，用于显示图片
6. CascadeClassifier.detectMultiScale() 方法，用于检测图片中的人脸
'''

import cv2
import os
import matplotlib.pyplot as plt

# 人脸检测的接口，这个是 OpenCV 中自带的
cascade_path = './Emotion_Recognition_File/face_detect_model/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)
'''
xml文件是OpenCV中的一个训练好的分类器，也就是一个训练好的模型，包含了所有必要的参数和特征，用于快速准确地检测图像中的面部。
'''

# img_path = "./Emotion_Recognition_File/face_det_img/" # 测试图片路径
img_path = "./opencv_facetest" # 

def DelNoneFaceImage(img_path):
    """
    删除没有人脸的图片
    """
    images = os.listdir(img_path)
    '''
    os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中。
    参数：path，表示要列出的目录
    返回值：返回指定路径下的文件和文件夹列表
    '''
    for image in images:
        im = cv2.imread(os.path.join(img_path, image), 1) # 读取图片
        rects = cascade.detectMultiScale(im, 1.3, 5)  # 人脸检测函数
        '''
        cascade.detectMultiScale()函数是OpenCV中的一个人脸检测函数，用于检测图片中的人脸。
        它接收三个参数：
          im：待检测的图像，通常是灰度图像，因为Haar特征基于灰度值计算。
          1.3：scaleFactor参数，指定在图像尺寸减小时，搜索窗口的缩放比例。
                值越大，检测速度越快，但可能错过一些小的或者是距离较远的面部。
          5：minNeighbors参数，指定每个候选矩形应该保留的邻近矩形的最小数量。
            这个参数控制着检测的质量，较高的值会减少检测到的假阳性，但也可能错过一些真正的对象。
        返回值是一个矩形列表，每个矩形表示一个检测到的人脸。
        '''
        print("检测到人脸的数量", len(rects))
        if len(rects) == 0:  # len(rects) 是检测人脸的数量，如果没有检测到人脸的话，会显示出图片，适合本地调试使用，在服务器上可能不会显示
            os.remove(os.path.join(img_path, image)) # 
            print("删除图片：", image)
            print()
    print("删除完成")
    return


def DetcFaceImage(img_path):
    """
    检测图片中的人脸，并显示出来
    """
    images = os.listdir(img_path)
    for image in images:
        im = cv2.imread(os.path.join(img_path, image), 1) # 读取图片
        rects = cascade.detectMultiScale(im, 1.3, 5)  # 人脸检测函数
        print("检测到人脸的数量", len(rects))
        if len(rects) == 0:  # len(rects) 是检测人脸的数量，如果没有检测到人脸的话，会显示出图片，适合本地调试使用，在服务器上可能不会显示
            pass
        plt.imshow(im[:, :, ::-1])  # 显示
        '''
        plt.imshow()函数用于显示图片，接收一个参数，即要显示的图片。
        由于OpenCV读取的图片是BGR格式，而matplotlib显示的图片是RGB格式，所以需要将BGR格式转换为RGB格式。
        [:, :, ::-1]表示将图片的第三个维度（颜色通道）反转，即将BGR转换为RGB。
        '''
        plt.show()
# DelNoneFaceImage(img_path)
DetcFaceImage(img_path)