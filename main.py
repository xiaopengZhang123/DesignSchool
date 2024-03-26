"""
created by zxp
@Date 2024/3/12
@Author zxp
"""
__author__ = 'zxp'

# 这是我们的图片文件路径

KNOWN_IMAGE_PATH = "./download.jpg"
TARGET1 = "./target1.jpg"
TARGET2 = "./target2.jpg"
TARGET3 = "./target3.jpg"

"""

"""


import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel,QGridLayout
from PyQt5.QtGui import QImage, QPixmap
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from serial.tools import list_ports

# 这是我们的目标图片
template_image = cv2.imread(KNOWN_IMAGE_PATH,cv2.IMREAD_GRAYSCALE)

# 先提取出来
template_image1 = cv2.imread(TARGET1,cv2.IMREAD_GRAYSCALE)
template_image2 = cv2.imread(TARGET2,cv2.IMREAD_GRAYSCALE)
template_image3 = cv2.imread(TARGET3,cv2.IMREAD_GRAYSCALE)



class PushButton(QWidget):
    def __init__(self):
        super(PushButton, self).__init__()
        self.search_chuankou = None
        self.match_thread = None
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
        self.show()


    def open_camera(self):
        # 打开默认的第一个摄像头
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("无法打开摄像头")
                sys.exit(-1)

            # 创建定时器，每隔一段时间获取一帧图像并显示
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.show_frame_and_search_template)
            # 每500毫秒获取一次新帧，根据实际情况调整
            self.timer.start(100)

    # c style name
    # 这里其实有两种方法，如果老师要问的话，可以使用模板匹配或者SIFT算法（特征提取）
    def show_frame_and_search_template(self):
        # 从摄像头中去读取我们的视频帧
        ret, frame = self.cap.read()
        if not ret:
            print("无法获取摄像头帧")
            sys.exit(-1)

        # 转换为灰度图像
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        matched_template_number = None

        for i, template_image in enumerate([template_image1, template_image2, template_image3]):
            method = cv2.TM_CCOEFF_NORMED

            # 执行模板匹配
            result = cv2.matchTemplate(gray_frame, template_image, method)
            # 寻找最大匹配值及其位置
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                match_location = min_loc
            else:
                match_location = max_loc

            # 设置匹配阈值
            match_threshold = 0.4

            if max_val >= match_threshold:
                print(f"目标已匹配！位置：({match_location[0]}, {match_location[1]}), 匹配到的是模板图片{i + 1}")
                matched_template_number = i + 1
                w, h = template_image.shape[::-1]
                cv2.rectangle(frame, match_location, (match_location[0] + w, match_location[1] + h), (0, 255, 0), 2)
                break  # 停止搜索其他模板，因为已经找到一个匹配项

        if matched_template_number is None:
            print("未找到任何目标匹配")

        # 显示结果图像
        cv2.imshow('Match Result', frame)

        # 等待用户按键，按任意键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.timer.stop()
            self.cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    def initWindow(self):
        self.setWindowTitle("PushButton")
        self.setGeometry(200, 200, 980, 540)

        self.openButton = QPushButton("打开摄像头", self)
        self.openButton.setShortcut('Ctrl+O')
        self.openButton.clicked.connect(self.open_camera)

        self.closeButton = QPushButton("关闭", self)
        self.closeButton.setShortcut('Ctrl+Q')
        self.closeButton.clicked.connect(self.close_camera)

        self.start_search_button = QPushButton("开始寻找", self)
        self.start_search_button.clicked.connect(self.startSearchTarget)

        self.search_chuankou = QPushButton("寻找串口", self)
        self.search_chuankou.clicked.connect(self.start_search_chuankou)

        layout = QGridLayout(self)
        layout.addWidget(self.openButton, 0, 0)
        layout.addWidget(self.closeButton, 0, 1)
        layout.addWidget(self.start_search_button, 1, 0)
        layout.addWidget(self.search_chuankou, 1, 1)

        self.is_searching = False  # 控制是否进行模板匹配的标志位

    def close_camera(self):
        if self.cap is not None:
            self.cap.release()  # 关闭摄像头
            self.timer.stop()  # 停止定时器
            self.cap = None  # 重置摄像头对象引用
            cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
            self.is_searching = False  # 清除搜索标志位

    def startSearchTarget(self):
        if not self.cap:
            print("未打开摄像头，不可以进行寻找")
        else:
            self.is_searching = True
            print("点击了按钮，现在开始寻找")

    def start_search_chuankou(self):
        port_list = list(list_ports.comports())
        num = len(port_list)

        if num <= 0:
            print("找不到任何串口设备")
        else:
            for i in range(num):
                port = list(port_list[i])
                print(port)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PushButton()
    sys.exit(app.exec_())
