'''
说明：
    class Cal_Interface 希望有的功能：
        将屏幕分割成小方格，随机将一定数量的小方格标记出来
        当用户在注视这些小方格的时候用摄像头捕获用户图像，保存下来
        作为标定图片

    2017年10月16日：
        完成：
            1.每隔 1s 标识出一个矩形框
        TODO：
            在摄像头中捕获用户的图像，并且保存
'''
import cv2
import numpy as np

class Cal_Interface(object):
    def __init__(self,sre_resolution=(1080,1920,3),
                 line_num=10,line_width=1,line_color=(255,255,255)
                 ,blockcolor=(250,180,222),blockwideth=5):
        '''
        初始化函数
        :param sre_resolution: 屏幕分辨率，默认为 （1080,1920,3）
        :param line_num:  图像画的横、竖线个数
        :param line_width:  线宽
        :param line_color:  线的颜色
        :param blockcolor:  矩形框的颜色
        :param blockwideth:  矩形框的宽度
        '''

        #默认分辨率为 （1080,1920）
        self.img_res=sre_resolution
        #保存画线的参数
        self.line_num=line_num  #线的数量
        self.line_width=line_width  #线宽
        self.line_color=line_color  #线颜色
        #保存方框格参数
        self.blockwideth=blockwideth  #线宽
        self.blockcolor=blockcolor  #线颜色

    def display_grid(self):
        '''
        @@@废弃，没有在类中调用@@@

        全屏显示图片,等待键盘输入任意值摧毁窗口
        :return:
        '''
        img_=self.cal_img
        cv2.namedWindow('Calibrate',cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Calibrate', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Calibrate',img_)
        cv2.waitKey()
        cv2.destroyWindow('Calibrate')

    def drawline(self,img_,line_w=1,line_color=(0,0,0),wandh_num=3):
        '''
        在图片上格子
        :param line_w: 线宽
        :param line_color: 线颜色
        :param wandh_num:  长宽线的数量
        :return:  无
        '''
        h,w=img_.shape[0],img_.shape[1]
        w_num,h_num=wandh_num,wandh_num
        h_,w_=h//h_num,w//w_num

        # 竖线 (w,h)
        for i in range(1,w_num):
            #print(i)
            cv2.line(img_,(w_*i,0),(w_*i,h),line_color,line_w)
        # 横线
        for i in range(1,h_num):
            cv2.line(img_,(0,h_*i),(w,h_*i),line_color,line_w)


    def drawblock(self,img,block_id=(3,3),blockcolor=(210,240,50),blockwideth=5):
        '''
        选定九宫格，在这个格子上填充矩形表示选定这个格子
        :param img_: 图片
        :param block: 九宫格序号
        :param blockcolor: 矩形框颜色
        :param blockwideth: 框的宽度
        :return:
        '''
        h,w=img.shape[0],img.shape[1]
        w_line,h_line=self.line_num,self.line_num
        h_,w_=h//h_line,w//w_line
        sx,sy=block_id[0]*w_,block_id[1]*h_

        #将整个矩形填充为其他颜色
        img[sy:sy+h_,sx:sx+w_,:]=blockcolor
        #在矩形中心画一个小圆辅助
        roi_=img[sy:sy+h_,sx:sx+w_]
        cv2.circle(roi_,(roi_.shape[1]//2,roi_.shape[0]//2), 10, (130,230,220), -1)

        #只是在矩形边缘画框
        #cv2.rectangle(img_,(sx,sy),(sx+w_,sy+h_),blockcolor,blockwideth)
        return img

    def cal(self,cal_times,wait_sec=1000):
        '''
        校准函数，随机显示若干个方格，捕获用户的图片，保存下来
        按 Esc 退出
        :param cal_times: 显示的方格个数
        :param wait_sec: 每个方格出现的时间间隙
        :return:
        '''

        #创建随机坐标
        x_index=np.random.choice(range(self.line_num),cal_times)
        y_index=np.random.choice(range(self.line_num),cal_times)
        index=zip(list(x_index),list(y_index))

        #设置window 为全屏
        cv2.namedWindow('Calibrate',cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Calibrate', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        #随机显示某个位置的方格，设置1 s后保存得到的图像

        for index_ in index:
            img_=np.ones(self.img_res,np.uint8)*128
            self.drawline(img_=img_,wandh_num=self.line_num,line_color=(255,255,255))
            self.drawblock(img=img_,block_id=index_,blockcolor=self.blockcolor)
            cv2.imshow('Calibrate',img_)
            #等待 1s 后保存图像
            if (cv2.waitKey(wait_sec) & 0xff==27):
                print('Exit Calibrate!')
                break
            #保存用户图像
            self.GetUserImage()
        cv2.destroyWindow('Calibrate')
        print('Get Calibrate Image Done!')

    def GetUserImage(self,label=None,savefile=None):
        '''
        获取用户图像并且保存
        :param label: 当前用户注视的坐标方向
        :param savefile: 保存路径
        :return: 无
        '''
        pass

if __name__ == '__main__':

    a=Cal_Interface(line_num=10)
    a.cal(100)
