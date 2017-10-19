'''
说明：
    class Cal_Interface 希望有的功能：
        将屏幕分割成小方格，随机将一定数量的小方格标记出来
        当用户在注视这些小方格的时候用摄像头捕获用户图像，保存下来
        作为标定图片

    2017年10月16日：
        完成：
            1.每隔 1s 标识出一个矩形框

            2.在摄像头中捕获用户的图像，并且保存

'''
import cv2
import numpy as np
import time
import os

class Cal_Interface(object):
    def __init__(self,sre_resolution=(1080,1920,3),cal_time=None,wait_sec=1000,
                 time_gap=20,frame_num=3,save_filename='calimg_file',
                 line_num=10,line_width=1,line_color=(255,255,255)
                 ,blockcolor=(250,180,222),blockwideth=5):
        '''
        初始化函数
        :param sre_resolution: 屏幕分辨率，默认为 （1080,1920,3）
        :param cal_time: 校正点的数量
        :param wait_sec: 当点显示后延时多久开始捕捉图像
        :param time_gap:每一帧图片之间的时间间隔
        :param frame_num:每个点捕捉的图像帧数
        :param line_num:  图像画的横、竖线个数
        :param line_width:  线宽
        :param line_color:  线的颜色
        :param blockcolor:  矩形框的颜色
        :param blockwideth:  矩形框的宽度
        '''

        #默认分辨率为 （1080,1920）
        self.img_res=sre_resolution
        #保存文件夹位置
        self.save_filename=save_filename
        #校正点的个数
        if cal_time is None:
            self.caltime=line_num*line_num
        else:
            self.caltime=cal_time
        #延时多久开始捕捉图像
        self.wait_sec=wait_sec
        #每一帧图片之间的时间间隔
        self.time_gap=time_gap
        #每个点捕捉的图像帧数
        self.framenum=frame_num
        #保存画线的参数
        self.line_num=line_num  #线的数量
        self.line_width=line_width  #线宽
        self.line_color=line_color  #线颜色
        #保存方框格参数
        self.blockwideth=blockwideth  #线宽
        self.blockcolor=blockcolor  #线颜色
        #摄像机capture
        self.cap=cv2.VideoCapture(0)

        #文件名与标签对应的字典


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


    def drawblock(self,img,block_id=0,blockcolor=(0,0,0),blockwideth=5):
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
        cor_h=block_id//self.line_num
        cor_w=block_id%self.line_num
        sx,sy=cor_w*w_,cor_h*h_

        #将整个矩形填充为其他颜色
        img[sy:sy+h_,sx:sx+w_,:]=blockcolor
        #在矩形中心画一个小圆辅助
        # roi_=img[sy:sy+h_,sx:sx+w_]
        # cv2.circle(roi_,(roi_.shape[1]//2,roi_.shape[0]//2), 10, (130,230,220), -1)

        #只是在矩形边缘画框
        #cv2.rectangle(img_,(sx,sy),(sx+w_,sy+h_),blockcolor,blockwideth)
        return img

    # def cal(self,cal_times,wait_sec=1000):
    #     '''
    #     校准函数，随机显示若干个方格，捕获用户的图片，保存下来
    #     按 Esc 退出
    #     :param cal_times: 显示的方格个数
    #     :param wait_sec: 每个方格出现的时间间隙
    #     :return:
    #     '''
    #
    #     #创建随机坐标
    #     x_index=np.random.choice(range(self.line_num),cal_times)
    #     y_index=np.random.choice(range(self.line_num),cal_times)
    #     index=zip(list(x_index),list(y_index))
    #
    #     #设置window 为全屏
    #     cv2.namedWindow('Calibrate',cv2.WINDOW_NORMAL)
    #     cv2.setWindowProperty('Calibrate', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #
    #     #随机显示某个位置的方格，设置1 s后保存得到的图像
    #
    #     for index_ in index:
    #         img_=np.ones(self.img_res,np.uint8)*128
    #         self.drawline(img_=img_,wandh_num=self.line_num,line_color=(255,255,255))
    #         self.drawblock(img=img_,block_id=index_,blockcolor=self.blockcolor)
    #         cv2.imshow('Calibrate',img_)
    #         #等待 1s 后保存图像
    #         if (cv2.waitKey(wait_sec) & 0xff==27):
    #             print('Exit Calibrate!')
    #             break
    #         #保存用户图像
    #         self.GetUserImage(savefile='calimg_file',label=index_)
    #     cv2.destroyWindow('Calibrate')
    #     print('Get Calibrate Image Done!')
    def cal(self,cal_times,wait_sec=1000):
        '''
        校准函数，随机显示若干个方格，捕获用户的图片，保存下来
        按 Esc 退出
        修改lable 为block id，也就是改预测模型为分类而不是回归
        不要设置 cal_times 参数
        :param cal_times: 显示的方格个数
        :param wait_sec: 每个方格出现的时间间隙
        :return:
        '''

        #创建乱序的block id
        block_index=np.arange(self.line_num*self.line_num)
        np.random.shuffle(block_index)

        #设置window 为全屏
        cv2.namedWindow('Calibrate',cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Calibrate', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        #随机显示某个位置的方格，设置1 s后保存得到的图像

        for index_ in block_index:
            img_=np.ones(self.img_res,np.uint8)*128
            self.drawline(img_=img_,wandh_num=self.line_num,line_color=(255,255,255))
            self.drawblock(img=img_,block_id=index_,blockcolor=self.blockcolor)
            cv2.imshow('Calibrate',img_)
            #等待 1s 后保存图像
            if (cv2.waitKey(wait_sec) & 0xff==27):
                print('Exit Calibrate!')
                break
            #保存用户图像
            self.GetUserImage(savefile=self.save_filename,label=index_)
        cv2.destroyWindow('Calibrate')
        print('Get Calibrate Image Done!')

    def GetUserImage(self,label=None,savefile='calimg_file',frame_num=10):
        '''
        获取用户图像并且保存
        一次调用 存储frame_num张图片，每张图片之间的时间间隙为frame_gap
        一共耗时 frame_gap*frame_num 毫秒
        :param label: 当前用户注视的坐标方向
        :param savefile: 保存路径
        :param frame_num: 每次坐标捕捉的帧数
        :return: 无
        '''
        cap=self.cap
        s_time=time.time()
        f_counter=0
        time_gap=self.time_gap
        #创建保存图片的文件夹
        if not os.path.exists(savefile):
            os.mkdir(savefile)
        #捕捉帧
        while (True):
            ret,fram=cap.read()
            if ret:
                f_counter+=1
                time_stamp=str(int(time.time()*1e7))
                #img_name=time_stamp+'_'+str(f_counter)+'_'+'x'+str(label[0])+'_y'+str(label[1])+'.jpg'
                #block id
                img_name=time_stamp+'_'+'blockid_'+str(label)+'.jpg'
                cv2.imwrite(os.path.join(savefile,img_name),fram)
                cv2.waitKey(time_gap)
            #30秒 超时退出
            if (time.time()-s_time>30):
                print('Some thing wrong with the cam,time out!')
                break
            if (f_counter>frame_num):
                #print('get user picture ok!')
                break
    def cameracheck(self):
        face_cascade = cv2.CascadeClassifier(r'D:\Proj_DL\Code\Proj_EyeTraker\haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(r'D:\Proj_DL\Code\Proj_EyeTraker\haarcascade_eye.xml')
        self.face_cascade=face_cascade
        self.eye_cascade=eye_cascade
        cap=self.cap
        s_time=time.time()
        print('start camera check!')
        while True:
            ret,frame=cap.read()
            if ret:
                self.drew_face_eye(frame)
                cv2.imshow('test',frame)
                if cv2.waitKey(30)&0xff == 27:
                    print('camera check ok!')
                    break
            rt=time.time()-s_time
            #超时退出
            if rt >60:
                print('camera check time out')
                break
        cv2.destroyWindow('test')

    def drew_face_eye(self,img):
        '''
        检测用户当前环境是否能够比较好的识别面部特征
        辨识出脸部以及眼部，标记出来
        :param img:
        :return:
        '''
        face_cascade=self.face_cascade
        eye_cascade=self.eye_cascade
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            #检测视频中脸部的眼睛，并用vector保存眼睛的坐标、大小（用矩形表示）
            eyes = eye_cascade.detectMultiScale(roi_gray,scaleFactor=1.3, minNeighbors=7, minSize=(50, 50),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
            for e in eyes:
                xe,ye,we,he=e
                cv2.rectangle(roi_color,(xe,ye),(xe+we,ye+he),(0,0,255),2)
        return  img

    def getoutclean(self):
        '''
        在结束标定的时候进行清理
        :return:
        '''
        self.cap.release()
        print('cam release!')
        cv2.destroyAllWindows()
        print('all windows destroy!')

    def starcalibrate(self):
        self.cameracheck()
        self.cal(self.caltime,wait_sec=self.wait_sec)

        self.getoutclean()

if __name__ == '__main__':

    a=Cal_Interface(line_num=4,save_filename='calimg_file_num')
    a.starcalibrate()
