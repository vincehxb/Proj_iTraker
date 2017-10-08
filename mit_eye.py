'''
images={'left_eye':left_eye,
        'right_eye':right_eye,
        'face_ori':face_ori
        'face_grid':face_grid}
'''
import tensorflow as tf
import numpy as np

class mit_itraker(object):
    def __init__(self,left_eye,right_eye,face_ori,face_grid):
        pass

    def _buildgraph(self,left_eye,right_eye,face_ori,face_grid):

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


    def conv_layer(self,x,name,shape):
        '''
        卷积层函数
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
        池化函数
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
            w=tf.get_variable('weight',shape=shape)
            b=tf.get_variable('biases',shape=shape[-1:])
            return tf.matmul(x,w)+b
