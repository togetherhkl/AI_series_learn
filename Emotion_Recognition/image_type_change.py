'''
该py文件用一个函数listfiles于将指定目录下的所有图片文件转换为jpg格式的图片文件并删除原图片文件。
参数：rootDir，表示要转换的图片文件所在的目录
返回值：无
总结：
1.使用os.walk()函数遍历指定目录下的所有文件和子目录。
2.使用cv2.imread()函数读取图片文件。
3.使用cv2.imwrite()函数保存图片文件。
4.使用os.remove()函数删除原图片文件。
'''

import os
import cv2
def listfiles(rootDir):
    list_dirs = os.walk(rootDir)
    '''
    用于生成文件夹中的文件名通过在目录树中游走。
    这个函数返回一个「三元组」(dirpath, dirnames, filenames)。
    dirpath是一个字符串，表示当前正在遍历的目录的路径。
    dirnames是一个列表，包含了dirpath下所有子目录的名字。
    filenames是一个列表，包含了非目录文件的名字。
    这行代码中的list_dirs将会是一个「迭代器」，每次迭代返回上述的三元组，
    表示rootDir目录及其所有子目录中的文件和目录信息。
    '''
    for root, dirs, files in list_dirs:
        for d in dirs:
            print(os.path.join(root, d))#打印目录
        for f in files:
            fileid = f.split('.')[0]#获取文件名
            filepath = os.path.join(root, f)#获取文件路径
            print(filepath)
            try:#异常处理，确保程序不会因为读取图片出错而中断
                src = cv2.imread(filepath,1)#读取图片
                '''
                cv2.imread()函数读取图片，接收两个参数
                第一个参数是图片路径，第二个参数是读取图片的方式（1表示读取彩色图片，参数0表示读取灰度图片）
                返回值是一个numpy数组，表示图片的像素矩阵。
                '''
                print("src=",filepath,src.shape)#src.shape是图片的尺寸
                os.remove(filepath)#删除原图片
                cv2.imwrite(os.path.join(root,fileid+".jpg"),src)#保存图片
                '''
                cv2.imwrite()函数保存图片，接收两个参数
                第一个参数是保存图片的路径，第二个参数是图片的像素矩阵
                返回值是一个布尔值，表示是否保存成功
                '''
            except:
                os.remove(filepath)
                continue
path = "./opencv_facetest"
listfiles(path)