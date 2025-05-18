import cv2 as cv
# 打印版本
print(cv.__version__)
import os

'''
# 1 图片操作
# 1.1 读取图片并展示图片
import sys
# img = cv.imread(cv.samples.findFile("OpenCV/test.jpg"))#使用imread函数读取图片
img = cv.imread("OpenCV/test.jpg")#使用imread函数读取图片
if img is None:
    sys.exit("Could not read the image.")

cv.imshow("Display window", img)
k = cv.waitKey(0)#等待键盘输入，0表示无限等待，1表示等待1ms

if k == ord("s"):
    cv.imwrite("starry_night.png", img)#保存图片，第一个参数是保存的文件名，第二个参数是图片对象
'''
'''
# 2 视频操作
# 2.1 读取视频并展示视频
import numpy as np

cap = cv.VideoCapture(0)#打开摄像头
# cap = cv.VideoCapture(cv.samples.findFile("OpenCV/test.mp4"))#打开视频文件，
# cap = cv.VideoCapture("OpenCV/test.mp4")#打开视频文件
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()#读取视频帧，ret表示是否读取成功，frame表示读取的视频帧
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)#将视频帧转换为灰度图
    # cv.imshow("frame", frame)#展示彩色图
    cv.imshow("frame", gray)#展示灰度图
    if cv.waitKey(1) == ord("q"):#按q退出
        break

cap.release()#释放摄像头
cv.destroyAllWindows()#关闭窗口
'''

'''
# 2.2 保存视频
cap = cv.VideoCapture(0)#打开摄像头

fourcc = cv.VideoWriter_fourcc(*"mp4v")#设置视频编码格式
out = cv.VideoWriter("output.mp4", fourcc, 25.0, (640, 480))#设置输出视频文件名，编码格式，帧率，分辨率

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.flip(frame, 1)#翻转视频帧


    out.write(frame)#写入视频帧
    cv.imshow("frame", frame)
    if cv.waitKey(1) == ord("q"):#等待1ms内按q退出
        break

cap.release()
out.release()
cv.destroyAllWindows()
'''

# 3 绘图
# 3.1 画线
import numpy as np
img = np.zeros((512, 512, 3), np.uint8)#创建一个512*512的全0数组
cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)#画一条蓝色线,起点(0,0)，终点(511,511)，线宽5

# 3.2 画矩形
cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)#画一个绿色矩形，左上角(384,0)，右下角(510,128)，线宽3
# 3.3 画圆
cv.circle(img, (447, 63), 63, (0, 0, 255), -1)#画一个红色圆，圆心(447,63)，半径63，线宽-1表示填充
# 3.4 画椭圆
cv.ellipse(img, (256, 256), (100, 50), 0, 0, 360, (255, 0, 0), -1)#画一个蓝色椭圆，中心(256,256)，长轴100，短轴50，角度0-180，线宽-1表示填充
# 3.5 画多边形
pts = np.array([[100, 200], [200, 100], [70, 200], [50, 100]], np.int32)#创建一个多边形
pts = pts.reshape((-1, 1, 2))#重塑数组
cv.polylines(img, [pts], False, (0, 255, 255))#画一个黄色多边形，pts表示多边形的顶点，True表示闭合多边形

cv.imshow("image", img)
k = cv.waitKey(4000)#等待键盘输入，0表示无限等待，1表示等待1ms

