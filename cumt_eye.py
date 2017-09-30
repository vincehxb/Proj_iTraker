import tensorflow as tf
import numpy as np

class Cumt_itraker(object):
    def __init__(self,image,dropout_rate,bn_flag):
        self._buildgraph(image,dropout_rate,bn_flag)
        print('Cumt_itraker Init Ready!')
    def _buildgraph(self,image,dropout,training):
        with tf.name_scope('Conv_Block_1'):
            conv1_=self.conv_layer(image,'conv1_1',[3,3,3,64],training)
            maxpool1_=self.maxpool_layer(conv1_,'pool_1')

        with tf.name_scope('Conv_Block_2'):
            conv2_=self.conv_layer(maxpool1_,'conv2_1',[3,3,64,64],training)
            conv3_=self.conv_layer(conv2_,'conv2_2',[3,3,64,64],training)
            maxpool2_=self.maxpool_layer(conv3_,'pool_2')

        with tf.name_scope('Conv_Block_3'):
            conv5_=self.conv_layer(maxpool2_,'conv3_1',[3,3,64,64],training)
            conv6_=self.conv_layer(conv5_,'conv3_2',[3,3,64,64],training)
            maxpool3_=self.maxpool_layer(conv6_,'pool_2')

        with tf.name_scope('FC_Block_4'):
            x_flatten=tf.reshape(maxpool3_,[-1,4*16*64])
            fc7=tf.nn.relu(self.fc_layer(x_flatten,'fc7',shape=[4*16*64,1000],dp=dropout))
            score=self.fc_layer(fc7,'fc8',shape=[1000,10],dp=dropout)
            self.score=score
    def conv_layer(self,x,name,shape,train_flag):
        with tf.variable_scope(name):
            w=tf.get_variable('weight',shape=shape,initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable('biases',shape=shape[-1:],initializer=tf.zeros_initializer())
            conv_=tf.nn.conv2d(x,w,[1,1,1,1],'SAME')
            bn_=tf.layers.batch_normalization(conv_,training=train_flag)
            relu_=tf.nn.relu(bn_)
            return relu_
    def maxpool_layer(self,x,name):
        with tf.variable_scope(name):
            return tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],'SAME',name='maxpool')
    def fc_layer(self,x,name,shape,dp):
        with tf.variable_scope(name):
            w=tf.get_variable('weight',shape=shape)
            b=tf.get_variable('biases',shape=shape[-1:])
            x=tf.nn.dropout(x,dp)
            return tf.matmul(x,w)+b



