'''
默认视频格式为 640*640，使用其他大小的视频需要对程序进行更改
'''
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
def drawline(img_,line_w=3,line_color=(255,255,255)):
    '''
    在图片上画九宫格
    :param img_:图片
    :param line_w: 线宽
    :param line_color: 线的颜色
    :return:
    '''
    cv2.line(img_,(213,0),(213,640),line_color,line_w)
    cv2.line(img_,(426,0),(426,640),line_color,line_w)
    cv2.line(img_,(0,213),(640,213),line_color,line_w)
    cv2.line(img_,(0,426),(640,426),line_color,line_w)
    return img_

def drawblock(img_,block,blockcolor=(210,240,50),blockwideth=5):
    '''
    选定九宫格，在这个格子上画矩形表示选定这个格子
    :param img_: 图片
    :param block: 九宫格序号
    :param blockcolor: 矩形框颜色
    :param blockwideth: 框的宽度
    :return:
    '''
    block_map={
        1:(0,0),2:(0,213),3:(0,426),
        4:(213,0),5:(213,213),6:(213,426),
        7:(426,0),8:(426,213),9:(426,426),
    }
    sy,sx=block_map[block]

    cv2.rectangle(img_,(sx,sy),(sx+213,sy+213),blockcolor,blockwideth)
    return img_

def drew_face_eye(img,minNeighbors=7,scaleFactor=1.1,minSize=(30, 30),reduce_pix=7):
    '''
    眼睛检测函数，使用opencv内置的API
    :param img:
    :return:
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    roi_color=None
    for (x,y,w,h) in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        #检测视频中脸部的眼睛，并用vector保存眼睛的坐标、大小（用矩形表示）
        eyes = eye_cascade.detectMultiScale(roi_gray,scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize,
                                            flags=cv2.CASCADE_SCALE_IMAGE)
        #眼睛检测 ,对于识别比较差的情况舍弃
        if len(eyes)!=2:
            return None
        if eyes[0][0]>eyes[1][0]:
            ex=eyes[1][0]
            W=eyes[0][0]-eyes[1][0]+eyes[0][2]
        else:
            ex=eyes[0][0]
            W=eyes[1][0]-eyes[0][0]+eyes[1][2]

        if eyes[0][1]>eyes[1][1]:
            ey=eyes[1][1]
            H=eyes[0][1]-eyes[1][1]+eyes[0][3]
        else:
            ey=eyes[0][1]
            H=eyes[1][1]-eyes[0][1]+eyes[1][3]

        cv2.rectangle(roi_color,(ex,ey),(ex+W,ey+H),(0,0,255),2)

    return roi_color[ey+reduce_pix:ey+H-reduce_pix,ex+reduce_pix:ex+W-reduce_pix] if roi_color is not None else None
