'''
该py文件用于获取人脸关键点，并根据关键点获取嘴唇区域。
思路：
1.使用Dlib库加载预训练的人脸关键点检测模型。
2.使用OpenCV加载人脸检测器。
3.遍历指定目录下的所有图片文件。
4.使用OpenCV读取图片。
5.使用Dlib检测人脸关键点。
6.根据关键点获取嘴唇区域。
7.保存嘴唇区域。
'''
import cv2
import dlib
import numpy as np
import os
import matplotlib.pyplot as plt

# 配置 Dlib 关键点检测路径
# 文件可以从 http://dlib.net/files/ 下载
PREDICTOR_PATH = "./Emotion_Recognition_File/face_detect_model/shape_predictor_68_face_landmarks.dat"
'''
预训练的人脸关键点检测模型，用于检测人脸的关键点。
'''
predictor = dlib.shape_predictor(PREDICTOR_PATH)
'''
dlib.shape_predictor()函数用于加载一个预训练的人脸关键点检测模型。
这个函数接收一个参数，表示模型文件的路径。
'''
# 配置人脸检测器路径
cascade_path = './Emotion_Recognition_File/face_detect_model/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

def get_landmarks(im):
    '''
    函数：获取人脸关键点
    参数：im，表示输入的图片
    返回值：返回一个矩阵，表示人脸关键点的坐标
    '''
    rects = cascade.detectMultiScale(im, 1.3, 5) # 人脸检测
    x, y, w, h = rects[0]  # 获取人脸的四个属性值，左上角坐标 x,y 、高宽 w、h
    #array([[209, 156, 289, 289]])
    #print(x, y, w, h)
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    '''
    目的：将人脸检测的结果转换为dlib的矩形对象
    dlib.rectangle()函数用于创建一个矩形对象，表示一个矩形区域。
    这个函数接收四个参数，分别是矩形的左上角和右下角的坐标。
    ''' 
    return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
    '''
    np.matrix()函数用于将列表转换为矩阵。
    predictor(im, rect)：这个函数调用使用predictor对象（之前通过dlib.shape_predictor加载的面部关键点检测器）
                        在图像im中的指定区域rect（一个矩形，通常是通过面部检测得到的）上检测面部关键点。
                        predictor函数返回一个包含检测到的面部关键点的对象。
    .parts()：这个方法被调用来获取检测到的所有面部关键点。它返回一个包含多个点的对象，每个点代表一个关键点的位置。
    [p.x, p.y for p in ...]：这是一个列表推导式，用于遍历parts()返回的所有关键点p，
                             并为每个关键点创建一个包含其x和y坐标的列表。
    np.matrix([...])：这将上一步得到的坐标列表转换成一个NumPy矩阵。NumPy矩阵是一个二维数组，
                     这里每一行代表一个关键点的x和y坐标
    '''


def annotate_landmarks(im, landmarks):
    '''
    目的：在图片上标记关键点
    参数：im，表示输入的图片
          landmarks，表示关键点坐标
    '''
    im = im.copy()# 复制图片
    for idx, point in enumerate(landmarks):
        #enumerate()函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im,
                    str(idx),# 文本内容
                    pos,# 位置
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,# 字体
                    fontScale=0.4,# 字体大小
                    color=(0, 0, 255))# 颜色
        cv2.circle(im, pos, 5, color=(0, 255, 255))# 画圆
    return im


def getlipfromimage(im, landmarks):
    '''
    功能：获取嘴唇区域
    参数：im，表示输入的图片
          landmarks，表示关键点坐标
    返回值：返回一个矩形区域，表示嘴唇区域
    '''
    xmin = 10000
    xmax = 0
    ymin = 10000
    ymax = 0
    # 根据最外围的关键点获取包围嘴唇的最小矩形框
    # 68 个关键点是从
    # 左耳朵0 -下巴-右耳朵16-左眉毛（17-21）-右眉毛（22-26）-左眼睛（36-41）
    # 右眼睛（42-47）-鼻子从上到下（27-30）-鼻孔（31-35）
    # 嘴巴外轮廓（48-59）嘴巴内轮廓（60-67）
    for i in range(48, 67):
        '''
        for循环遍历关键点，找到嘴唇区域的最小矩形框
        '''
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

    print("xmin=", xmin)
    print("xmax=", xmax)
    print("ymin=", ymin)
    print("ymax=", ymax)

    roiwidth = xmax - xmin# 嘴唇区域的宽度
    roiheight = ymax - ymin# 嘴唇区域的高度

    roi = im[ymin:ymax, xmin:xmax, 0:3]
    '''
    im[ymin:ymax, xmin:xmax, 0:3]：这是一个NumPy数组切片操作，用于从原始图像中提取嘴唇区域。
    它接收三个参数，分别是ymin:ymax、xmin:xmax和0:3。
    ymin:ymax和xmin:xmax表示切片的范围，0:3表示提取所有的通道（RGB）。
    该操作返回一个包含嘴唇区域的NumPy数组。
    '''

    # 将嘴唇区域扩大1.5倍
    if roiwidth > roiheight:
        dstlen = 1.5 * roiwidth
    else:
        dstlen = 1.5 * roiheight
    # 计算嘴唇区域的中心点
    diff_xlen = dstlen - roiwidth
    diff_ylen = dstlen - roiheight

    newx = xmin
    newy = ymin

    imagerows, imagecols, channel = im.shape
    '''
    im.shape是一个元组，包含三个元素，分别是图像的行数、列数和通道数。
    '''
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
        newy = imagerows - dstlen
    '''
    上述代码：通过智能调整ROI的位置，确保ROI始终位于图像的有效区域内，
    既不会超出图像边界，也尽可能保持其原始尺寸和形状。
    这在进行图像处理和分析时非常重要，特别是在需要精确操作图像特定区域的应用中
    '''
    roi = im[int(newy):int(newy + dstlen), int(newx):int(newx + dstlen), 0:3]
    '''
    使用了Python的切片语法来从图像中提取ROI。
    int(newy):int(newy + dstlen)和int(newx):int(newx + dstlen)分别定义了ROI在y方向和x方向上的范围。
    0:3指定了颜色通道的范围，对于一个标准的BGR颜色图像，这意味着选择所有三个颜色通道（蓝色、绿色、红色）
    '''
    return roi


def listfiles(rootDir):
    list_dirs = os.walk(rootDir)
    for root, dirs, files in list_dirs:
        for d in dirs:
            print(os.path.join(root, d))
        for f in files:
            fileid = f.split('.')[0]

            filepath = os.path.join(root, f)
            try:
                im = cv2.imread(filepath, 1)
                landmarks = get_landmarks(im)# 获取关键点
                roi = getlipfromimage(im, landmarks)# 获取嘴唇区域
                roipath = filepath.replace('.jpg', '_mouth.png')# 保存嘴唇区域
#                 cv2.imwrite(roipath, roi)
                plt.imshow(roi[:, :, ::-1])
                plt.show()
            except:
#                 print("error")
                continue

listfiles("./Emotion_Recognition_File/face_det_img/")