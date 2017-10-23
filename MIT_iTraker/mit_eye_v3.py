'''
    说明：
        version 3 主要区别是增加直接加载 NPY权值文件的功能，同时去除BN层
                  主要目的是为了将原来的回归问题转变成分类问题，因为前面的模型保存原因不能够直接在原模型上修改，所以另外建立了一个新版本

    修改：
        2017年10月19日10:34:57
            1.去除BN层
            2.添加加载权值函数
            3.权值初始化文件需要调用 mit_itraker 内的初始化函数，不要直接初始化
'''
import tensorflow as tf
import numpy as np
import os
class mit_itraker(object):
    def __init__(self,fileroot_addr,left_eye,right_eye,face_ori,dropout,reg_rate,sess,load_trainimg=True,
                 output_class=16,weight_fileaddr=None):

        self.regularizer=tf.contrib.layers.l2_regularizer(reg_rate)

        self.output_class=output_class
        self.dropout=dropout
        self.sess=sess

        self._buildgraph(left_eye,right_eye,face_ori,dropout)

        #如果有设置权值文件路径的话就加载权值
        if weight_fileaddr:
            self.weight_fileaddr=weight_fileaddr
        #是否需要加载MIT训练图片集
        if load_trainimg:
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

    def reset_fclayer(self,sess,init_newclasslayer=False):
        '''
        重置最后两层 FC1，FC2的权值，适应新的数据集
        :return:
        '''
        with tf.variable_scope('FC_Concate'):
            with tf.variable_scope('FC_1',reuse=True):
                w=tf.get_variable('weight',shape=(256, 128))
                b=tf.get_variable('biases',shape=(128,))
                wfc1=tf.truncated_normal([256,128],stddev=0.01)
                bfc1=tf.zeros([128,])
                sess.run(w.assign(wfc1))
                sess.run(b.assign(bfc1))
        with tf.variable_scope('FC_Concate'):
            with tf.variable_scope('FC_2',reuse=True):
                w=tf.get_variable('weight',shape=(128, 2))
                b=tf.get_variable('biases',shape=(2,))
                wfc2=tf.truncated_normal([128,2],stddev=0.01)
                bfc2=tf.zeros([2,])
                sess.run(w.assign(wfc2))
                sess.run(b.assign(bfc2))
        print('Reset FC1,FC2 weight/biases done!')
        if init_newclasslayer:
            with tf.variable_scope('FC_2_CLASS',reuse=True):
                w=tf.get_variable('FC_2/weight',shape=[128,self.output_class])
                b=tf.get_variable('FC_2/biases',shape=[self.output_class,])
                wfc3=tf.truncated_normal([128,self.output_class],stddev=0.01)
                bfc3=tf.zeros([self.output_class,])
                sess.run(w.assign(wfc3))
                sess.run(b.assign(bfc3))
            print('Init FC_2_CLASS weight/biases done!')

    def _initweight(self):
        weightfileaddr=self.weight_fileaddr
        sess=self.sess
        weight_dict=np.load(weightfileaddr)
        init=tf.global_variables_initializer()
        sess.run(init)
        print('converting weight to the layer !')
        #加载CONV 层 W,b
        for i in range(1,5):
            # CONV_1 -- CONV_4
            block_name='CONV_'+str(i)
            with tf.variable_scope(block_name,reuse=True):
                # conv_E1 -- conv_E4 , conv_F1 -- conv_F4
                for j in ['E','F']:
                    layer_name='conv_'+j+str(i)
                    w_=tf.get_variable(layer_name+r'/weight')
                    b_=tf.get_variable(layer_name+r'/biases')
                    sess.run(w_.assign(weight_dict[layer_name]['weight']))
                    sess.run(b_.assign(weight_dict[layer_name]['biases']))
                    
        with tf.variable_scope('FC_Concate',reuse=True):
            fce1_w=tf.get_variable('FC_E1/weight')
            fce1_b=tf.get_variable('FC_E1/biases')
            sess.run(fce1_w.assign(weight_dict['FC_E1']['weight']))
            sess.run(fce1_b.assign(weight_dict['FC_E1']['biases']))
            fcf1_w=tf.get_variable('FC_F1/weight')
            fcf1_b=tf.get_variable('FC_F1/biases')
            sess.run(fcf1_w.assign(weight_dict['FC_F1']['weight']))
            sess.run(fcf1_b.assign(weight_dict['FC_F1']['biases']))
            #fc1_w=tf.get_variable('FC_1/weight')
            #fc1_b=tf.get_variable('FC_1/biases')
            #sess.run(fc1_w.assign(weight_dict['FC_1']['weight']))
            #sess.run(fc1_b.assign(weight_dict['FC_1']['biases']))
        print(r'weight/biases init done!')

    def _buildgraph(self,left_eye,right_eye,face_ori,dropout_rate):

        '''
        创建网络结构
        :param left_eye: 左眼数据
        :param right_eye: 右眼数据
        :param face_ori:  脸部数据
        :param face_grid:  脸部掩码
        :param dropout_rate:  dropout的概率值
        :return: 各分类的分数

        '''
        output_class=self.output_class
        #left_eye,right_eye,face_ori,face_grid=self.LeftEye,self.RightEye,self.FaceOri,self.FaceMask

        with tf.variable_scope('CONV_1'):       #   64*64/3 -> 54*54/96 -> 27*27/96
            #  left eyes conv->relu->maxpool

            left_conv1=self.conv_layer(left_eye,'conv_E1',[11,11,3,96])
            left_pool1=self.maxpool_layer(left_conv1,'left_pool1')

            #  right eyes conv->relu->maxpool

            right_conv1=self.conv_layer(right_eye,'conv_E1',[11,11,3,96],reuse_=True,)
            right_pool1=self.maxpool_layer(left_conv1,'right_pool1')

            #  face conv->relu->maxpool

            face_conv1=self.conv_layer(face_ori,'conv_F1',[11,11,3,96])
            face_pool1=self.maxpool_layer(face_conv1,'face_pool1')

        with tf.variable_scope('CONV_2'):       #   27*27/96 -> 23*23/256 -> 10*10/256
            #  left eyes conv->relu->maxpool

            left_conv2=self.conv_layer(left_pool1,'conv_E2',[5,5,96,256])
            left_pool2=self.maxpool_layer(left_conv2,'left_pool2')

            #  right eyes conv->relu->maxpool

            right_conv2=self.conv_layer(right_pool1,'conv_E2',[5,5,96,256],reuse_=True)
            right_pool2=self.maxpool_layer(right_conv2,'right_pool2')

            #  face conv->relu->maxpool

            face_conv2=self.conv_layer(face_pool1,'conv_F2',[5,5,96,256])
            face_pool2=self.maxpool_layer(face_conv2,'face_pool2')

        with tf.variable_scope('CONV_3'):       #   10*10/256 -> 8*8/384 -> 4*4/384
            #  left eyes conv->relu->maxpool

            left_conv3=self.conv_layer(left_pool2,'conv_E3',[3,3,256,384])
            left_pool3=self.maxpool_layer(left_conv3,'left_pool3')

            #  right eyes conv->relu->maxpool

            right_conv3=self.conv_layer(right_pool2,'conv_E3',[3,3,256,384],reuse_=True)
            right_pool3=self.maxpool_layer(right_conv3,'right_pool3')

            #  face conv->relu->maxpool

            face_conv3=self.conv_layer(face_pool2,'conv_F3',[3,3,256,384])
            face_pool3=self.maxpool_layer(face_conv3,'face_pool3')

        with tf.variable_scope('CONV_4'):       #   4*4/384 -> 4*4/64 -> 2*2/64
            #  left eyes conv->relu->maxpool

            left_conv4=self.conv_layer(left_pool3,'conv_E4',[1,1,384,64])
            left_pool4=self.maxpool_layer(left_conv4,'left_pool4')

            #  right eyes conv->relu->maxpool

            right_conv4=self.conv_layer(right_pool3,'conv_E4',[1,1,384,64],reuse_=True)
            right_pool4=self.maxpool_layer(right_conv4,'right_pool4')

            #  face conv->relu->maxpool

            face_conv4=self.conv_layer(face_pool3,'conv_F4',[1,1,384,64])
            face_pool4=self.maxpool_layer(face_conv4,'face_pool4')

        with tf.variable_scope('FC_Concate'):
            # concate left and right eyes
            x_leftEye=tf.reshape(left_pool4,shape=[-1,2*2*64],name='leftEye_flatten') # N*256
            x_rightEye=tf.reshape(right_pool4,shape=[-1,2*2*64],name='rightEye_flatten') # N*256
            eyes_concate=tf.concat([x_leftEye,x_rightEye],axis=1,name='eyes_concate') # N*512
            FC_E1=self.fc_layer(eyes_concate,'FC_E1',[512,128],dp=dropout_rate)# N*128

            # concate face and face_mask
            #x_faceMask=tf.reshape(face_grid,shape=[-1,25*25],name='facemask_flatten') # N*625
            x_face=tf.reshape(face_pool4,shape=[-1,2*2*64],name='face_flatten') # N*256
            FC_F1=self.fc_layer(x_face,name='FC_F1',shape=[2*2*64,128],dp=dropout_rate) # N*128
            #FC_FG1=self.fc_layer(x_faceMask,name='FC_FG1',shape=[25*25,256],dp=dropout_rate,bn_flag=bn_train) # N*256
            #faces_concate=tf.concat([FC_F1,FC_FG1],axis=1,name='faces_concate') # N*384

            # concate faces and eyes
            face_eye_concate=tf.concat([FC_F1,FC_E1],axis=1,name='face_eye_concate') # N*512
            FC_1=self.fc_layer(face_eye_concate,'FC_1',[256,1024],dp=dropout_rate) # N*128
            FC_1=tf.nn.relu(FC_1,'fc1_relu')
            #FC_2=self.fc_layer(FC_1,'FC_2',[128,2],dp=dropout_rate) # N*2
            # 转换成分类问题
            FC2_CLASS=self.fc_layer(FC_1,'FC2_CLASS',[1024,output_class],dp=dropout_rate)
        self.score=FC2_CLASS
        print('Building graph done!')


    def conv_layer(self,x,name,shape,reuse_=None):
        '''
        卷积层函数，默认无 padding,初始化为  xavier
        加入BN层
        :param x:输入X
        :param name: 卷积层名称
        :param bn_flag: True表示训练，False表示测试
        :return: 卷积+BN+relu后的结果
        '''
        if name[-1]=='4' or name[-1]=='3':
            trainable_=True
        else:
            trainable_=False
        with tf.variable_scope(name,reuse=reuse_):
            w=tf.get_variable('weight',shape=shape,initializer=tf.contrib.layers.xavier_initializer(),trainable=trainable_)
            b=tf.get_variable('biases',initializer=tf.constant(0.1,shape=shape[-1:]))
            tf.add_to_collection('loss_w',self.regularizer(w))
            conv_=tf.nn.conv2d(x,w,[1,1,1,1],'VALID')+b

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
            tf.add_to_collection('loss_w',self.regularizer(w))
            fc_op=tf.matmul(x,w)+b

            return fc_op
