'''
默认视频格式为 640*640，使用其他大小的视频需要对程序进行更改
'''
import cv2
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
