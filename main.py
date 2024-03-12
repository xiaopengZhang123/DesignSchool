"""
created by zxp
@Date 2024/3/12
@Author zxp
"""
__author__ = 'zxp'

# 这是我们的图片文件路径

KNOWN_IMAGE_PATH = "./download.jpg"

"""
1. 首先是我们的一些思路
    我们首先要用PyQt来创建一个窗口
    然后这个窗口上面要有启动摄像头的功能
    然后这个摄像头还要起到的作用是利用SIFT算法来进行图像的图片
    这里为了简化目标,我们可以先在控制台打印一些信息

2.其次就是我们的开发框架
    我们先需要去构建一个UI界面(后续可以美化,先从简单的来)
    然后就是需要去摄像头流里面去打印
    多线程???
    线程安全的问题要考虑吗???
    应该不会发生争抢的情况吧
    
3.目前的问题
    视频把我们的按键给屏蔽了,需要我们点击好拍照之后才可以启动摄像头(已解决)
    还需要的就是阻塞问题,目前主进程是被开始摄像头这个进程所阻塞的,所以如果还需要发送消息的话,必须得使用并发多线程得方式(未解决)
    也就是异步非阻塞得方式去收到这条信号
    然后这条信号,我们就可以去发送一些东西...
    难道还需要一个进程?
    或者说,如果我们直接识别到了目标图片,当前进程得父进程直接使用waitpid函数把子进程给回收了吧???
    
"""

import time
from PyQt5.uic.properties import QtGui
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox
import numpy as np
import cv2
import time
from random import uniform
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
import cv2
from PyQt5.QtCore import pyqtSlot, QTimer, Qt
from PyQt5.QtCore import QThread, pyqtSignal, QTimer


class PushButton(QWidget):
    def __init__(self):
        super(PushButton, self).__init__()
        self.MIN_MATCH_COUNT = 10
        self.start_search_button = None
        self.timer = None
        self.closeButton = None
        self.cap = None
        self.openButton = None
        self.video_frame = QLabel(self)
        # 初始化窗口
        self.initWindow()
        # 初始化布局
        self.create_layout()
        self.show()

    def open_camera(self):
        # 打开默认的第一个摄像头
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
            # 创建定时器，每隔一段时间获取一帧图像并显示
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.show_frame)
            # 每30毫秒获取一次新帧，根据实际情况调整
            # 这个其实也不需要去设置这么快
            self.timer.start(30)

    @pyqtSlot()
    def show_frame(self):
        ret, frame = self.cap.read()  # 读取摄像头的一帧图像

        if ret:
            # 转换图像格式以适应Qt
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 设置长宽高
            h, w, ch = img.shape
            bytes_per_line = ch * w
            Qtformat = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(Qtformat)
            self.video_frame.setPixmap(pixmap.scaled(self.video_frame.size(), Qt.KeepAspectRatio))

        else:
            self.cap.release()
            self.timer.stop()
            print("无法从摄像头获取数据")

    def initWindow(self):
        # 设置窗口信号
        self.setWindowTitle("PushButton")
        # 设置大小
        # 长 宽 高 低
        self.setGeometry(200, 200, 980, 540)

        # 创建并配置“打开摄像头”按钮
        self.openButton = QPushButton(self)
        self.openButton.setText("打开摄像头")
        # 加个快捷键
        self.openButton.setShortcut('Ctrl+O')
        # 当按钮被点击时，连接open_camera槽函数
        self.openButton.clicked.connect(self.open_camera)
        self.openButton.move(900, 10)  # 左上角坐标为(10, 10)

        # 创建一个关闭摄像头
        self.closeButton = QPushButton(self)
        self.closeButton.setText("关闭")  # text
        self.closeButton.setShortcut('Ctrl+Q')
        # self.closeButton.move(900,500)

        # 当按钮被点击时，连接open_camera槽函数
        self.closeButton.clicked.connect(self.close_camera)

        # 创建开始寻找的按钮
        # 还是先创建了吧,方便调试一点
        self.start_search_button = QPushButton();
        self.start_search_button.setText("开始寻找")
        self.start_search_button.clicked.connect(self.startSearchTarget);

    def create_layout(self):
        layout = QVBoxLayout()
        # 添加窗口按钮
        layout.addWidget(self.openButton)
        # 关闭按钮
        layout.addWidget(self.closeButton)
        # 开启摄像头框架
        layout.addWidget(self.video_frame)
        # 开始寻找图片
        layout.addWidget(self.start_search_button)
        self.setLayout(layout)

    def close_camera(self):
        if self.cap is not None:
            self.cap.release()  # 关闭摄像头
            self.timer.stop()  # 停止定时器
            self.cap = None  # 重置摄像头对象引用
            self.video_frame.clear()  # 清空视频帧显示区域
        else:
            pass

    def startSearchTarget(self):
        # 加载已知图片并准备匹配所需数据
        target_image = cv2.imread(KNOWN_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
        sift = cv2.xfeatures2d.SIFT_create()
        target_keypoints, target_descriptors = sift.detectAndCompute(target_image, None)

        # 创建匹配线程
        match_thread = ImageMatchThread(self, target_image, sift, target_keypoints, target_descriptors, self.cap)
        match_thread.match_found_signal.connect(self.on_match_found)
        match_thread.start()

    def on_match_found(self):
        print("目标图片已在视频流中找到！")


class ImageMatchThread(QThread):
    match_found_signal = pyqtSignal()

    def __init__(self, parent, target_image, sift, target_keypoints, target_descriptors, cap):
        super(ImageMatchThread, self).__init__(parent)
        self.target_image = target_image
        self.sift = sift
        self.target_keypoints = target_keypoints
        self.target_descriptors = target_descriptors
        self.cap = cap

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_keypoints, frame_descriptors = self.sift.detectAndCompute(gray_frame, None)

            matcher = cv2.BFMatcher()
            matches = matcher.match(self.target_descriptors, frame_descriptors)

            good_matches = [m for m in matches if m.distance < 0.75 * min([m.distance for m in matches])]

            if len(good_matches) > self.parent().MIN_MATCH_COUNT:
                self.match_found_signal.emit()
                # 发现目标后，可以考虑停止线程以节省资源，也可以继续循环监控
                # break


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PushButton()
    sys.exit(app.exec_())
