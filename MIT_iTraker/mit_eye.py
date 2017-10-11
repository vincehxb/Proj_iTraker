'''
2017年10月8日22:46:03
    完成：
        1._buildgraph函数
        2.conv、maxpool、fc的包装函数

2017年10月9日14:56:51
    完成：
        1.不在类内定义占位符
        2.在类内加载数据，并对数据进行normalize

2017年10月10日20:37:49
    完成：
        1.在fc层加入 dropout
        2.加入BatchNormalize层
        3.loss里面加入W的惩罚项


'''
import tensorflow as tf
import numpy as np
import os
class mit_itraker(object):
    def __init__(self,fileroot_addr,left_eye,right_eye,face_ori,face_grid,dropout,bn_train,reg_rate):

        self.regularizer=tf.contrib.layers.l2_regularizer(reg_rate)

        self._buildgraph(left_eye,right_eye,face_ori,face_grid,dropout,bn_train)
        print('Loading file~')
        self._loadnpz(fileroot_addr)

    def _loadnpz(self,file_addr):
        '''
        * 加载图片文件，将所有加载的图，坐标放入 data字典中
        * 同时对数据进行 normalize使得均值为 0，值域为 1
        :param file_addr: 文件夹地址
        :return: 装有所有图片以及坐标的字典data
        '''
        self.data={}
        mean_dict={}
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
            ori_shape=d_.shape
            d_=d_.reshape(ori_shape[0],-1).astype('float32')
            if i[-5]!='y':
                #使得所有值小于 1
                d_/=255.0
                #使得均值为0
                mean_=np.mean(d_,axis=0)
                d_-=mean_
                mean_dict[i[:-4]]=mean_
            self.data[i[:-4]]=d_.reshape(ori_shape)
        self.mean_dict=mean_dict
        print('Load Data Done~')

    def _buildgraph(self,left_eye,right_eye,face_ori,face_grid,dropout_rate,bn_train):

        '''
        创建网络结构
        :param left_eye: 左眼数据
        :param right_eye: 右眼数据
        :param face_ori:  脸部数据
        :param face_grid:  脸部掩码
        :param dropout_rate:  dropout的概率值
        :return: 回归的坐标

        '''

        #left_eye,right_eye,face_ori,face_grid=self.LeftEye,self.RightEye,self.FaceOri,self.FaceMask

        with tf.variable_scope('CONV_1'):       #   64*64/3 -> 54*54/96 -> 27*27/96
            #  left eyes conv->relu->maxpool

            left_conv1=self.conv_layer(left_eye,'conv_E1',[11,11,3,96],bn_train)
            left_pool1=self.maxpool_layer(left_conv1,'left_pool1')

            #  right eyes conv->relu->maxpool

            right_conv1=self.conv_layer(right_eye,'conv_E1',[11,11,3,96],bn_train,reuse_=True,)
            right_pool1=self.maxpool_layer(left_conv1,'right_pool1')

            #  face conv->relu->maxpool

            face_conv1=self.conv_layer(face_ori,'conv_F1',[11,11,3,96],bn_train)
            face_pool1=self.maxpool_layer(face_conv1,'face_pool1')

        with tf.variable_scope('CONV_2'):       #   27*27/96 -> 23*23/256 -> 10*10/256
            #  left eyes conv->relu->maxpool

            left_conv2=self.conv_layer(left_pool1,'conv_E2',[5,5,96,256],bn_train)
            left_pool2=self.maxpool_layer(left_conv2,'left_pool2')

            #  right eyes conv->relu->maxpool

            right_conv2=self.conv_layer(right_pool1,'conv_E2',[5,5,96,256],bn_train,reuse_=True)
            right_pool2=self.maxpool_layer(right_conv2,'right_pool2')

            #  face conv->relu->maxpool

            face_conv2=self.conv_layer(face_pool1,'conv_F2',[5,5,96,256],bn_train)
            face_pool2=self.maxpool_layer(face_conv2,'face_pool2')

        with tf.variable_scope('CONV_3'):       #   10*10/256 -> 8*8/384 -> 4*4/384
            #  left eyes conv->relu->maxpool

            left_conv3=self.conv_layer(left_pool2,'conv_E3',[3,3,256,384],bn_train)
            left_pool3=self.maxpool_layer(left_conv3,'left_pool3')

            #  right eyes conv->relu->maxpool

            right_conv3=self.conv_layer(right_pool2,'conv_E3',[3,3,256,384],bn_train,reuse_=True)
            right_pool3=self.maxpool_layer(right_conv3,'right_pool3')

            #  face conv->relu->maxpool

            face_conv3=self.conv_layer(face_pool2,'conv_F3',[3,3,256,384],bn_train)
            face_pool3=self.maxpool_layer(face_conv3,'face_pool3')

        with tf.variable_scope('CONV_4'):       #   4*4/384 -> 4*4/64 -> 2*2/64
            #  left eyes conv->relu->maxpool

            left_conv4=self.conv_layer(left_pool3,'conv_E4',[1,1,384,64],bn_train)
            left_pool4=self.maxpool_layer(left_conv4,'left_pool4')

            #  right eyes conv->relu->maxpool

            right_conv4=self.conv_layer(right_pool3,'conv_E4',[1,1,384,64],bn_train,reuse_=True)
            right_pool4=self.maxpool_layer(right_conv4,'right_pool4')

            #  face conv->relu->maxpool

            face_conv4=self.conv_layer(face_pool3,'conv_F4',[1,1,384,64],bn_train)
            face_pool4=self.maxpool_layer(face_conv4,'face_pool4')

        with tf.variable_scope('FC_Concate'):
            # concate left and right eyes
            x_leftEye=tf.reshape(left_pool4,shape=[-1,2*2*64],name='leftEye_flatten') # N*256
            x_rightEye=tf.reshape(right_pool4,shape=[-1,2*2*64],name='rightEye_flatten') # N*256
            eyes_concate=tf.concat([x_leftEye,x_rightEye],axis=1,name='eyes_concate') # N*512
            FC_E1=self.fc_layer(eyes_concate,'FC_E1',[512,128],dp=dropout_rate,bn_flag=bn_train)# N*128

            # concate face and face_mask
            x_faceMask=tf.reshape(face_grid,shape=[-1,25*25],name='facemask_flatten') # N*625
            x_face=tf.reshape(face_pool4,shape=[-1,2*2*64],name='face_flatten') # N*256
            FC_F1=self.fc_layer(x_face,name='FC_F1',shape=[2*2*64,128],dp=dropout_rate,bn_flag=bn_train) # N*128
            FC_FG1=self.fc_layer(x_faceMask,name='FC_FG1',shape=[25*25,256],dp=dropout_rate,bn_flag=bn_train) # N*256
            faces_concate=tf.concat([FC_F1,FC_FG1],axis=1,name='faces_concate') # N*384

            # concate faces and eyes
            face_eye_concate=tf.concat([faces_concate,FC_E1],axis=1,name='face_eye_concate') # N*512
            FC_1=self.fc_layer(face_eye_concate,'FC_1',[512,128],dp=dropout_rate,bn_flag=bn_train) # N*128
            FC_1=tf.nn.relu(FC_1,'fc1_relu')
            FC_2=self.fc_layer(FC_1,'FC_2',[128,2],dp=dropout_rate,bn_flag=bn_train,output_flag=True) # N*2
        self.coordinate=FC_2
        print('Building graph done!')

    def conv_layer(self,x,name,shape,bn_flag,reuse_=None):
        '''
        卷积层函数，默认无 padding,初始化为  xavier
        加入BN层
        :param x:输入X
        :param name: 卷积层名称
        :param bn_flag: True表示训练，False表示测试
        :return: 卷积+BN+relu后的结果
        '''

        with tf.variable_scope(name,reuse=reuse_):
            w=tf.get_variable('weight',shape=shape,initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable('biases',initializer=tf.constant(0.1,shape=shape[-1:]))
            tf.add_to_collection('loss_w',self.regularizer(w))
            conv_=tf.nn.conv2d(x,w,[1,1,1,1],'VALID')
            conv_=tf.layers.batch_normalization(conv_,training=bn_flag)
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
    def fc_layer(self,x,name,shape,dp,bn_flag,output_flag=False):
        '''
        全连接层函数,加入dropout选项
        :param x: 输入
        :param name: 层名称
        :param dp: dropout选项,None表示不加
        :param bn_flag: True表示训练，False表示测试
        :param output_flag: True表示为最后一层FC，False表示不是最后一层FC.  最后一层FC不加BN,DP
        :return:  X*W+B 没有relu
        '''
        with tf.variable_scope(name):

            x=tf.nn.dropout(x,dp,name='dropout')
            w=tf.get_variable('weight',shape=shape,initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable('biases',initializer=tf.constant(0.1,shape=shape[-1:]))
            tf.add_to_collection('loss_w',self.regularizer(w))
            fc_op=tf.matmul(x,w)+b
            if not output_flag:
                fc_op=tf.layers.batch_normalization(fc_op,training=bn_flag)
            return fc_op
