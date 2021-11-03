import base64
import json
import os
import sys
from typing import Container

import cv2
import numpy as np
import requests
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QWidget

#修改数据库连接，并进行数据库迁移操作

class Ui_MainWindow(QtWidgets.QWidget):

    def __init__(self,parent=None):

        super().__init__(parent) #父类的构造函数

        self.timer_camera = QtCore.QTimer() #定义定时器，用于控制显示视频的帧率

        self.cap = cv2.VideoCapture()       #视频流

        self.CAM_NUM = 0                    #为0时表示视频流来自笔记本内置摄像头

        self.set_ui()                       #初始化程序界面

        self.slot_init()                    #初始化槽函数

 
    '''程序界面布局'''

    def set_ui(self):

        self.__layout_main = QtWidgets.QHBoxLayout()           #总布局

        self.__layout_fun_button = QtWidgets.QVBoxLayout()      #按键布局

        self.__layout_data_show = QtWidgets.QVBoxLayout()       #数据(视频)显示布局

        self.button_open_camera = QtWidgets.QPushButton('打开相机') #建立用于打开摄像头的按键

        self.button_close = QtWidgets.QPushButton('退出')           #建立用于退出程序的按键

        self.button_open_camera.setMinimumHeight(50)                #设置按键大小

        self.button_close.setMinimumHeight(50)

        self.setWindowTitle('袋鼯麻麻——智能购物平台')
        self.setWindowIcon(QIcon('./image/icon.png'))

        self.button_close.move(10,100)                      #移动按键

        '''信息显示'''

        self.label_show_camera = QtWidgets.QLabel()   #定义显示视频的Label

        self.label_show_camera.setFixedSize(641,481)    #给显示视频的Label设置大小为641x481

        '''把按键加入到按键布局中'''

        self.__layout_fun_button.addWidget(self.button_open_camera) #把打开摄像头的按键放到按键布局中

        self.__layout_fun_button.addWidget(self.button_close)       #把退出程序的按键放到按键布局中

        '''把某些控件加入到总布局中'''

        self.__layout_main.addLayout(self.__layout_fun_button)      #把按键布局加入到总布局中

        self.__layout_main.addWidget(self.label_show_camera)        #把用于显示视频的Label加入到总布局中

        '''总布局布置好后就可以把总布局作为参数传入下面函数'''

        self.setLayout(self.__layout_main) #到这步才会显示所有控件

 
    '''初始化所有槽函数'''

    def slot_init(self):

        self.button_open_camera.clicked.connect(self.button_open_camera_clicked)    #若该按键被点击，则调用button_open_camera_clicked()

        self.timer_camera.timeout.connect(self.show_camera) #若定时器结束，则调用show_camera()

        self.button_close.clicked.connect(self.close)#若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序


    '''槽函数之一'''

    def button_open_camera_clicked(self):

        if self.timer_camera.isActive() == False:   #若定时器未启动

            flag = self.cap.open(self.CAM_NUM) #参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频

            if flag == False:       #flag表示open()成不成功

                msg = QtWidgets.QMessageBox.warning(self,'warning',"请检查相机于电脑是否连接正确",buttons=QtWidgets.QMessageBox.Ok)

            else:

                self.timer_camera.start(30)  #定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示

                self.button_open_camera.setText('识别')

        else:

            self.timer_camera.stop()  #关闭定时器

            self.cap.release()        #释放视频流

            self.label_show_camera.clear()  #清空视频显示区域

            self.button_open_camera.setText('打开相机')

            self.container_recognition()


    def lable_close(self):

        if self.timer_camera.isActive():

            self.timer_camera.stop()

        if self.cap.isOpened():

            self.cap.release()

        self.label_show_camera.clear()


    def show_camera(self):

        flag,self.image = self.cap.read()  #从视频流中读取

        show = cv2.resize(self.image,(640,480))     #把读到的帧的大小重新设置为 640x480

        show = cv2.cvtColor(show,cv2.COLOR_BGR2RGB) #视频色彩转换回RGB，这样才是现实的颜色

        showImage = QtGui.QImage(show.data,show.shape[1],show.shape[0],QtGui.QImage.Format_RGB888) #把读取到的视频数据变成QImage形式

        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  #往显示视频的Label里 显示QImage


    def deplay(self, value):

        '''显示对话框返回值'''
        if value == "Please connect root to upload container's name and it's price!\n":
        
            QMessageBox.information(self, "Warning","{}".format(value), QMessageBox.Yes | QMessageBox.No)
        else:
            QMessageBox.information(self, "价格清单","您一共购买了：\n{}".format(value), QMessageBox.Yes | QMessageBox.No)

        self.lable_close()

        self.message = " No face_recognition"

        self.button_open_camera.setText('打开相机')


    def getByte(self, path):

        with open(path, 'rb') as f:

            img_byte = base64.b64encode(f.read()) #二进制读取后变base64编码

        img_str = img_byte.decode('ascii') #转成python的unicode

        return img_str 
    

    def container_recognition(self):

        self.picture_file = '.\\test_client_pic\\test_client_pic.jpg'

        cv2.imwrite(self.picture_file, self.image)

        img_str = self.getByte('.\\test_client_pic\\test_client_pic.jpg')

        requestsss={'name':'测试图片', 'image':img_str}
        req = json.dumps(requestsss) #字典数据结构变json(所有程序语言都认识的字符串)

        res=requests.post('localhost/reference_client/', data=req)
        print(type(res.text))
        json_res = json.loads(res.text)
        print(json_res['container'])
        container_all = json_res['container']
        if container_all =="Please connect root to upload container's name and it's price!\n":
            rec_deplay_str_all = container_all
        else:
            price_all = json_res['price_all']
            rec_docs_price_all = []
            
            for i in range(len(container_all)):
                rec_docs_price = []
                if i%2 == 0:
                    container = container_all[i]
                    price = container_all[i+1]
                    rec_docs_price.append(container)
                    rec_docs_price.append(price)
                    rec_docs_price_all.append(rec_docs_price)

            rec_deplay_str = ''

            for rec_single in rec_docs_price_all:
                rec_name = rec_single[0]
                rec_price = rec_single[1]
                rec_deplay_str = '商品：{}'.format(rec_name) + '\t' + '单价：{}元'.format(str(rec_price)) + '\n' + rec_deplay_str
                rec_deplay_str_all = rec_deplay_str + '\n' + '您需付款：{}元'.format(str(price_all))

        self.deplay(rec_deplay_str_all)


if __name__ =='__main__':

    app = QtWidgets.QApplication(sys.argv)  #固定的，表示程序应用

    ui = Ui_MainWindow()                    #实例化Ui_MainWindow

    ui.show()                               #调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的

    sys.exit(app.exec_())                   #不加这句，程序界面会一闪而过


