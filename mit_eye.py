'''
2017年10月8日22:46:03
    完成：
        1._buildgraph函数
        2.conv、maxpool、fc的包装函数
'''
import tensorflow as tf
import numpy as np
import os
class mit_itraker(object):
    def __init__(self,fileroot_addr):

        #self._defPlaceholder()
        self._buildgraph()
        self._loadnpz(fileroot_addr)


    def _defPlaceholder(self):
        '''
        定义占位符
        :return:
        '''
        with tf.name_scope('PlaceHolder'):
            self.LeftEye=tf.placeholder(tf.float32,[None,64,64,3],name='LeftEye')
            self.RightEye=tf.placeholder(tf.float32,[None,64,64,3],'RightEye')
            self.FaceOri=tf.placeholder(tf.float32,[None,64,64,3],'FaceOri')
            self.FaceMask=tf.placeholder(tf.float32,[None,25,25],'FaceMask')
            self.Y=tf.placeholder(tf.float32,[None,2],name='Y')
            self.LR=tf.placeholder(tf.float32,name='Learning_rate')
        print('Define PlaceHolder Done~')

    def _defTrainernLoss(self):
        self.LOSS=tf.nn.softmax_cross_entropy_with_logits()

    def _loadnpz(self,file_addr):
        '''
        * 加载图片文件，将所有加载的图，坐标放入 data字典中
        * 同时对数据进行 normalize使得均值为 0，值域为 1
        :param file_addr: 文件夹地址
        :return: 装有所有图片以及坐标的字典data
        '''
        self.data={}
        for i in ['train_eye_left.npy',
                  'train_eye_right.npy',
                  'train_face.npy',
                  'train_face_mask.npy',
                  'train_y.npy',
                  'val_eye_left.npy',
                  'val_eye_right.npy',
                  'val_face.npy',
                  'val_face_mask.npy',
                  'val_y.npy']:
            d_=np.load(os.path.join(file_addr,i))
            d_=d_.reshape(d_.shape[0],-1).astype('float32')
            #使得所有值小于 1
            d_/=255.0
            #使得均值为0
            d_-=np.mean(d_,axis=0)
        print('Load Data Done~')

    def _buildgraph(self):

        '''
        创建网络结构
        :param left_eye: 左眼数据
        :param right_eye: 右眼数据
        :param face_ori:  脸部数据
        :param face_grid:  脸部掩码
        :return: 回归的坐标
        '''

        left_eye,right_eye,face_ori,face_grid=self.LeftEye,self.RightEye,self.FaceOri,self.FaceMask

        with tf.variable_scope('CONV_1'):       #   64*64/3 -> 54*54/96 -> 27*27/96
            #  left eyes conv->relu->maxpool

            left_conv1=self.conv_layer(left_eye,'conv_E1',[11,11,3,96])
            left_pool1=self.maxpool_layer(left_conv1,'left_pool1')

            #  right eyes conv->relu->maxpool

            right_conv1=self.conv_layer(right_eye,'conv_E1',[11,11,3,96])
            right_pool1=self.maxpool_layer(left_conv1,'right_pool1')

            #  face conv->relu->maxpool

            face_conv1=self.conv_layer(face_ori,'conv_F1',[11,11,3,96])
            face_pool1=self.maxpool_layer(face_conv1,'face_pool1')

        with tf.variable_scope('CONV_2'):       #   27*27/96 -> 23*23/256 -> 10*10/256
            #  left eyes conv->relu->maxpool

            left_conv2=self.conv_layer(left_pool1,'conv_E2',[5,5,96,256])
            left_pool2=self.maxpool_layer(left_conv2,'left_pool2')

            #  right eyes conv->relu->maxpool

            right_conv2=self.conv_layer(right_pool1,'conv_E2',[5,5,96,256])
            right_pool2=self.maxpool_layer(right_conv2,'right_pool2')

            #  face conv->relu->maxpool

            face_conv2=self.conv_layer(face_pool1,'conv_F2',[5,5,96,256])
            face_pool2=self.maxpool_layer(face_conv2,'face_pool2')

        with tf.variable_scope('CONV_3'):       #   10*10/256 -> 8*8/384 -> 4*4/384
            #  left eyes conv->relu->maxpool

            left_conv3=self.conv_layer(left_pool2,'conv_E3',[3,3,256,384])
            left_pool3=self.maxpool_layer(left_conv3,'left_pool3')

            #  right eyes conv->relu->maxpool

            right_conv3=self.conv_layer(right_pool2,'conv_E3',[3,3,256,384])
            right_pool3=self.maxpool_layer(right_conv3,'right_pool3')

            #  face conv->relu->maxpool

            face_conv3=self.conv_layer(face_pool2,'conv_F3',[3,3,256,384])
            face_pool3=self.maxpool_layer(face_conv3,'face_pool3')

        with tf.variable_scope('CONV_4'):       #   4*4/384 -> 4*4/64 -> 2*2/64
            #  left eyes conv->relu->maxpool

            left_conv4=self.conv_layer(left_pool3,'conv_E4',[1,1,384,64])
            left_pool4=self.maxpool_layer(left_conv4,'left_pool4')

            #  right eyes conv->relu->maxpool

            right_conv4=self.conv_layer(right_pool3,'conv_E4',[1,1,384,64])
            right_pool4=self.maxpool_layer(right_conv4,'right_pool4')

            #  face conv->relu->maxpool

            face_conv4=self.conv_layer(face_pool3,'conv_F4',[1,1,384,64])
            face_pool4=self.maxpool_layer(face_conv4,'face_pool4')

        with tf.variable_scope('FC_Concate'):
            # concate left and right eyes
            x_leftEye=tf.reshape(left_pool4,shape=[-1,2*2*64],name='leftEye_flatten') # N*256
            x_rightEye=tf.reshape(right_pool4,shape=[-1,2*2*64],name='rightEye_flatten') # N*256
            eyes_concate=tf.concat([x_leftEye,x_rightEye],axis=1,name='eyes_concate') # N*512
            FC_E1=self.fc_layer(eyes_concate,'FC_E1')# N*128

            # concate face and face_mask
            x_faceMask=tf.reshape(face_grid,shape=[-1,25*25],name='facemask_flatten') # N*625
            x_face=tf.reshape(face_pool4,shape=[-1,2*2*64],name='face_flatten') # N*256
            FC_F1=self.fc_layer(x_face,name='FC_F1',shape=[2*2*64,128]) # N*128
            FC_FG1=self.fc_layer(x_faceMask,name='FC_FG1',shape=[25*25,256]) # N*256
            faces_concate=tf.concat([FC_F1,FC_FG1],axis=1,name='faces_concate') # N*384

            # concate faces and eyes
            face_eye_concate=tf.concat([faces_concate,FC_E1],axis=1,name='face_eye_concate') # N*512
            FC_1=self.fc_layer(face_eye_concate,'FC_1',[512,128]) # N*128
            FC_1=tf.nn.relu(FC_1,'fc1_relu')
            FC_2=self.fc_layer(FC_1,'FC_2',[128,2]) # N*2
        self.coordinate=FC_2
        print('Building graph done!')

    def conv_layer(self,x,name,shape):
        '''
        卷积层函数，默认无 padding,初始化为  xavier
        :param x:输入X
        :param name: 卷积层名称
        :return: 卷积+relu后的结果
        '''
        with tf.variable_scope(name):
            w=tf.get_variable('weight',shape=shape,initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable('biases',shape=shape[-1:],initializer=tf.constant(0.1,shape=shape[-1:]))
            conv_=tf.nn.conv2d(x,w,[1,1,1,1],'VALID')
            return tf.nn.relu(conv_)

    def maxpool_layer(self,x,name):
        '''
        池化函数，默认 K=2，S=2，也就是输入【H,W】,输出 【H/2,W/2】
        :param x:输入
        :param name: 层名字
        :return: 池化后的结果
        '''
        with tf.variable_scope(name):
            return tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],'VALID',name='maxpool')
    def fc_layer(self,x,name,shape):
        '''
        全连接层函数
        :param x: 输入
        :param name: 层名称
        :return:  X*W+B 没有relu
        '''
        with tf.variable_scope(name):
            w=tf.get_variable('weight',shape=shape,initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable('biases',shape=shape[-1:],initializer=tf.constant(0.1,shape=shape[-1:]))
            return tf.matmul(x,w)+b
