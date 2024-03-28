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

import numpy as np
import serial
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QGridLayout
import cv2
from PyQt5.QtCore import QTimer
from serial.tools import list_ports

# 这是我们的目标图片，这个就只是个测试文件罢了，测试一下这个template match的功能
template_image1 = cv2.imread(KNOWN_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

# 先提取出来
# 转换为灰度图
template_image2 = cv2.imread(TARGET1, cv2.IMREAD_GRAYSCALE)
template_image3 = cv2.imread(TARGET2, cv2.IMREAD_GRAYSCALE)
template_image4 = cv2.imread(TARGET3, cv2.IMREAD_GRAYSCALE)


class PushButton(QWidget):
    def __init__(self):
        super(PushButton, self).__init__()
        self.send_message = None
        self.find_chuankou = None
        self.search_chuankou = None
        self.match_thread = None
        self.MIN_MATCH_COUNT = 10
        self.start_search_button = None
        self.timer = None
        self.closeButton = None
        self.cap = None
        self.openButton = None
        # 设置串口那个，要不然还得进行异常处理
        self.video_frame = QLabel(self)
        # 初始化窗口
        self.initWindow()
        # 初始化布局
        self.show()

    def open_camera(self):
        # 打开默认的第一个摄像头
        if not self.cap:
            # 这里就是我们的默认摄像头位置
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                print("无法打开摄像头")
                sys.exit(-1)

            # 创建定时器，每隔一段时间获取一帧图像并显示
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.show_frame_and_search_template)
            # self.timer.timeout.connect(self.sift_match)

            # 每100毫秒获取一次新帧，根据实际情况调整
            self.timer.start(200)

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
        # 数字转换为列表，这里没有使用target1是因为target1是test文件
        for i, template_image in enumerate([template_image2, template_image3, template_image4]):
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
            # emmm... 这个阈值真是奇怪，目前还算可以匹配得到，后面就不知道了
            match_threshold = 0.3

            # 此代码是抄的画框部分
            if max_val >= match_threshold:
                print(f"目标已匹配！位置：({match_location[0]}, {match_location[1]}), 匹配到的是模板图片{i + 1}")
                matched_template_number = i + 1
                w, h = template_image.shape[::-1]
                cv2.rectangle(frame, match_location, (match_location[0] + w, match_location[1] + h), (0, 255, 0), 2)
                # 停止搜索其他模板，因为已经找到一个匹配项，找到就没必要去寻找其他的
                break
        # 没想好写什么
        if matched_template_number is None:
            pass

        # 显示结果图像
        cv2.imshow('Result', frame)

        # 等待用户按键，按任意键退出
        # 可有可无
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.timer.stop()
            self.cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    def initWindow(self):
        self.setWindowTitle("PushButton")
        self.setGeometry(200, 200, 980, 540)
        # 我觉得是函数指针
        # 加了一些快捷键，我觉得没啥大用
        self.openButton = QPushButton("打开摄像头", self)
        self.openButton.setShortcut('Ctrl+O')
        self.openButton.clicked.connect(self.open_camera)
        self.closeButton = QPushButton("关闭", self)
        self.closeButton.setShortcut('Ctrl+Q')
        self.closeButton.clicked.connect(self.close_camera)

        # 开始寻找按钮
        self.start_search_button = QPushButton("开始寻找", self)
        self.start_search_button.clicked.connect(self.startSearchTarget)

        # 寻找一下串口
        self.search_chuankou = QPushButton("寻找串口", self)
        self.search_chuankou.clicked.connect(self.start_search_chuankou)

        # 通过串口发送一些信息，我觉得后期可以把这个给封装一下
        # 这个信息多了去了，比如识别到图案会怎么样，识别不到又会怎么样
        # 而且还得考虑一下在什么时候去发送我们的信息
        self.send_message = QPushButton("发送信息", self);
        self.send_message.clicked.connect(self.open_chuankou)

        # 这个布局的排列方式还挺有意思的，还可以用matrix的形式
        layout = QGridLayout(self)
        layout.addWidget(self.openButton, 0, 0)
        layout.addWidget(self.closeButton, 0, 1)
        layout.addWidget(self.start_search_button, 1, 0)
        layout.addWidget(self.search_chuankou, 1, 1)
        layout.addWidget(self.send_message, 2, 0)

        # 控制是否进行模板匹配的标志位
        # 一开始肯定不去进行匹配啊，但是为了简单，直接把这个干到检测摄像头里面去了。。。 :(
        self.is_searching = False

        # 如果没有找到串口，那么就不可以发送信息
        # 防小人不防君子啊
        self.find_chuankou = False

    def close_camera(self):
        if self.cap is not None:
            # 关闭摄像头
            self.cap.release()
            # 停止定时器
            self.timer.stop()
            # 重置摄像头对象引用
            # 大抵每次都是要重新刷新一次
            self.cap = None
            # 关闭所有OpenCV窗口
            cv2.destroyAllWindows()
            self.is_searching = False  # 清除搜索标志位

    def startSearchTarget(self):
        print("Just A Test Message for startSearchTarget")
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
            return False
        else:
            for i in range(num):
                port_info = port_list[i]
                print(f"串口名称：{port_info.device}, 描述：{port_info.description}, 接口：{port_info.interface}")
                # 控制是否进行模板匹配的标志位
                self.find_chuankou = True

    # 打开串口通信
    def open_chuankou(self):
        # 定义串口参数
        # 替换成实际的串口号,可能是COM1
        if not self.find_chuankou:
            print("没有找到串口，你不可以发送信息")
        else:
            # 第二个口应该是串口7

            port = "COM7"
            # 波特率
            # 发送者的应与接收者的一样
            # 这里是多少还得问一下 bzr
            baudrate = 9600
            # 打开串口
            ser = serial.Serial(port, baudrate, timeout=1)
            # 发个0x3试一试，转换为字节
            data_to_send = bytes([0x3])
            # 发送数据
            ser.write(data_to_send)
            # 关闭串口
            # 有开有关
            ser.close()


    # backup 备用算法，不过在实现上还有点问题，还是某个cv库不太对劲
    # 这是sift算法的部分，这个算法更加适配
    def sift_match(self):
        # 从摄像头中读取视频帧
        ret, frame = self.cap.read()
        if not ret:
            print("无法获取摄像头帧")
            sys.exit(-1)

        # # 将模板图像和当前帧转换为灰度图像
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        templates = [cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY) for template_image in
                     [template_image1, template_image2, template_image3, template_image4]]

        for i, template in enumerate(templates):
            # 初始化SIFT检测器和匹配器
            sift = cv2.xfeatures2d.SIFT_create()

            # 提取模板图像和当前帧的SIFT特征点
            template_kp, template_des = sift.detectAndCompute(template, None)
            frame_kp, frame_des = sift.detectAndCompute(gray_frame, None)

            # 使用BFMatcher进行匹配
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(template_des, frame_des, k=2)

            # 应用 Lowe's 置信度比值测试进行有效匹配筛选
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            MIN_MATCH_COUNT = 0.4
            # 如果有足够的有效匹配，则计算对象的位置
            if len(good_matches) > MIN_MATCH_COUNT:  # MIN_MATCH_COUNT 是一个设定的阈值
                src_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                h, w = template.shape[:2]

                # 计算并绘制匹配区域
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                # 绘制矩形框
                img2 = cv2.polylines(frame, [np.int32(dst)], True, 255, 2, cv2.LINE_AA)

                print(f"目标已匹配！位置：({int(dst[0][0][0])}, {int(dst[0][0][1])}), 匹配到的是模板图片{i + 1}")

            # 显示结果图像
        cv2.imshow('Match Result', frame)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PushButton()
    sys.exit(app.exec_())
