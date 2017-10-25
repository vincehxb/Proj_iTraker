'''
2017年10月23日21:11:38
    实验：
        利用比较浅的 inception 结构来跑MIT的数据集
'''
import tensorflow as tf
import numpy as np
import cv2

class inception(object):
    def __init__(self):
        pass
    def incepA_block(self,incep_base,block_name,base_shape,reduce_size):
        with tf.variable_scope(block_name):
            base_1=self.conv_layer(incep_base,'reduce_a',[1,1,base_shape[-1],reduce_size])
    def stem_block(self):
        pass
    def conv_layer(self,x,name,shape,reuse_=None,filter_shape=[1,1,1,1],padding='VALID'):
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
            #tf.add_to_collection('loss_w',self.regularizer(w))
            conv_=tf.nn.conv2d(x,w,filter_shape,padding)+b

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
    def fc_layer(self,x,name,shape,dp):
        '''
        全连接层函数,加入dropout选项
        :param x: 输入
        :param name: 层名称
        :param dp: dropout选项,None表示不加
        :param bn_flag: True表示训练，False表示测试
        :return:  X*W+B 没有relu
        '''
        with tf.variable_scope(name):

            x=tf.nn.dropout(x,dp,name='dropout')
            w=tf.get_variable('weight',shape=shape,initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable('biases',initializer=tf.constant(0.1,shape=shape[-1:]))
            #tf.add_to_collection('loss_w',self.regularizer(w))
            fc_op=tf.matmul(x,w)+b

            return fc_op

