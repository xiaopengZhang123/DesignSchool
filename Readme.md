# 函数用法
1. 最重要的就是connect函数,其中语法如下
```python
self.控件对象名称.信号名称.connect(self.槽信号)
例如,我们这里有一个名字叫做button的控件
```
2. 首先是我们的一些思路
    - 我们首先要用PyQt来创建一个窗口
    - 然后这个窗口上面要有启动摄像头的功能
    - 然后这个摄像头还要起到的作用是利用SIFT算法来进行图像的图片
    - 这里为了简化目标,我们可以先在控制台打印一些信息

3. 其次就是我们的开发框架
   - 我们先需要去构建一个UI界面(后续可以美化,先从简单的来)
   - 然后就是需要去摄像头流里面去打印
   - 多线程???
   - 线程安全的问题要考虑吗???
   - 应该不会发生争抢的情况吧
    
4. 目前的问题
    - 视频把我们的按键给屏蔽了,需要我们点击好拍照之后才可以启动摄像头(已解决)
    - 还需要的就是阻塞问题,目前主进程是被开始摄像头这个进程所阻塞的,所以如果还需要发送消息的话,必须得使用并发多线程得方式(未解决)
    - 也就是异步非阻塞得方式去收到这条信号
    - 然后这条信号,我们就可以去发送一些东西...
    - 难道还需要一个进程?
    - 或者说,如果我们直接识别到了目标图片,当前进程得父进程直接使用waitpid函数把子进程给回收了吧???
    