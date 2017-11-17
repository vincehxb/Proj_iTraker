import tensorflow as tf
import numpy as np

class bisic_cnn(object):
    def __init__(self,sess):
       pass
    def _convlayer(self,x,name,shape,padding='VALID',stride=[1,1]):
        '''
        conv函数
        :param x:输入
        :param name: 模块名称
        :param shape: [filter_h,filter_w,input_channel,output_channel,]
        :param stride:卷积的h,w的步长
        :param padding:
        :return:
        '''
        with tf.variable_scope(name):
            w=tf.get_variable('weight',shape=shape,initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable('biases',shape=shape[-1:],initializer=tf.constant_initializer(0.01))
            conv_=tf.nn.conv2d(x,w,[1,stride[0],stride[1],1],padding=padding,name='conv')
            conv_=tf.nn.bias_add(conv_,b,name='bias_add')
            return conv_
    def _poollayer(self, input_x, pooling='max',size=(3, 3), stride=(2, 2), padding='SAME'):
        '''
        pool函数
        :param input:输入
        :param pooling: 'avg'表示均值池化，用于最后输出score.'max'表示maxpool
        :param size:
        :param stride:
        :param padding:
        :return:
        '''
        if pooling == 'avg':
            x = tf.nn.avg_pool(input_x, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                               padding=padding)
        else:
            x = tf.nn.max_pool(input_x, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                               padding=padding)
        return x

    def _convblock(self,x,name,shape,bn_istraing,padding='VALID',stride=[1,1]):
        '''
        标准的：CONV->BN -> RELU
        三个函数结合
        :param x:输入
        :param name: 模块名称
        :param shape: [filter_h,filter_w,input_channel,output_channel,]
        :param bn_istraing:是否是在训练
        :param stride:卷积的h,w的步长
        :param padding:
        :return:
        '''
        with tf.variable_scope(name):
            #conv
            with tf.name_scope('weight_biases'):
                w=tf.get_variable('weight',shape=shape,initializer=tf.contrib.layers.xavier_initializer())
                b=tf.get_variable('biases',shape=shape[-1:],initializer=tf.constant_initializer(0.01))
            conv_=tf.nn.conv2d(x,w,[1,stride[0],stride[1],1],padding=padding,name='conv')
            conv_=tf.nn.bias_add(conv_,b,name='bias_add')
            #BN
            bn_=tf.layers.batch_normalization(conv_,training=bn_istraing,name='BN')
            #relu
            relu_=tf.nn.relu(bn_,name='relu')
            return relu_

    def save_network_weight(self,filename,sess):
        '''
        提取网络的权值，并保存到文件
        :param filename: 文件名
        :return:
        '''
        import pickle
        print('extracting network weight to file:{}'.format(filename))
        all_var_name=list(tf.trainable_variables())
        weight_dict={}
        #提取变量值
        for name_ in all_var_name:
            #变量名称
            layer_name=str(name_).split("'")[1][:-2]
            print(layer_name)
            with tf.variable_scope('',reuse=True):
                var=tf.get_variable(layer_name)
                #注意var是tensor，需要转换一下
                weight_dict[layer_name]=sess.run(var)
        #保存到pkl文件中
        fp=open(filename,'wb')
        pickle.dump(obj=weight_dict,file=fp)
        fp.close()
        print('save weight file done!')
    def init_network(self,weight_addr,sess,skip_layer=[]):
        '''
        利用保存好的权值文件来初始化网络
        ！！注意假如需要调用这个函数，一定不可以在调用这个函数后在对全部变量进行初始化 （sess.run(init)）！！
        这样会使得加载的权值被覆盖，要先进行全局变量初始化再读取权值文件！
        :param weight_addr:权值文件
        :return:
        '''
        #初始化全部变量
        init=tf.global_variables_initializer()
        sess.run(init)
        #加载权值文件，写入网络
        print('loading weight file:{}'.format(weight_addr))
        network_dict=np.load(weight_addr)
        layer_name=list(network_dict.keys())
        for name_ in layer_name:
            if name_ in skip_layer:
                print('skip layer:{}'.format(name_))
                continue
            with tf.variable_scope('',reuse=True):
                var=tf.get_variable(name_)
                sess.run(var.assign(network_dict[name_]))
        print('network init done!')