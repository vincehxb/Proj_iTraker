'''
    更新：
        2017年11月12日20:39:06
            1.完成前向运行 predict函数，加载训练好的权值后可以直接调用这个函数进行图像预测
            2.例子：
                X=tf.placeholder(shape=[None,128,128,3],dtype=tf.float32)
                model=squeezenet(X,sess,outputclass=16)
                model.init_network('squeeze1k4x4_9688.pkl',skip_layer=[])
                e=model.geteyeimg(frame)
                asw_prob=model.predict(X_tensor=X,eyes_images=e)


        2017年11月10日16:50:17
            1.更新提取权值函数：init_network(weight_addr)
            2.更新保存权值函数：save_network_weight(filename)

    说明：
        2017年11月8日10:14:24
            利用SqueezeNet思想来重构CUMT结构，原CUMT网络参数大小为 5.26Mb(conv:0.57Mb,fc:4.68Mb)
            改进后的squeezenet的参数大小（conv:2.53Mb,fc:0Mb)

            网络结构：
            输入：（128,128,3）

            head:(N,128,128,3)->(N,64,64,64)
                conv1(64@3*3/1)->conv2(64@3*3/1)->maxpool

            fire_1:(N,64,64,64)->(N,32,32,128)
                squeeze(16@1*1/1) -> expand(32@3*3/1)   -> squeeze(16@1*1/1) -> expand(32@3*3/1)  -> squeeze(16@1*1/1) -> expand(64@3*3/1)  -> maxpool
                                  -> expand(32@1*1/1)                        -> expand(32@1*1/1)                       -> expand(64@1*1/1)

            fire_2:(N,32,32,128)->(N,16,16,384)
                squeeze(32@1*1/1) -> expand(64@3*3/1)   -> squeeze(32@1*1/1) -> expand(64@3*3/1)  -> squeeze(32@1*1/1) -> expand(192@3*3/1)  -> maxpool
                                  -> expand(64@1*1/1)                        -> expand(64@1*1/1)                       -> expand(192@1*1/1)

            fire_3:(N,16,16,384)->(N,8,8,512)
                squeeze(64@1*1/1) -> expand(192@3*3/1)   -> squeeze(64@1*1/1) -> expand(192@3*3/1)  -> squeeze(64@1*1/1) -> expand(256@3*3/1)  -> maxpool
                                  -> expand(192@1*1/1)                        -> expand(192@1*1/1)                       -> expand(256@1*1/1)

            tail:(N,8,8,512)->(N,10)
                dropout(0.5)->conv1(10@1*1/1)->avgpool
'''
import tensorflow as tf
import numpy as np
import pickle
import cv2
class squeezenet():
    def __init__(self,images,sess,lr=1e-3,imageshape=[128,128,3],outputclass=10):
        '''
        初始化：建立计算图
        :param images:
        :param sess:
        :param lr:
        :param imageshape:
        '''

        self.sess=sess
        print('building graph')
        self.outputclass=outputclass
        self._buildgraph(images)

        #眼部，脸部cascade
        self.face_cascade = None
        self.eye_cascade = None

    def _buildgraph(self,x,dropout_rate=0.5):
            '''
            建立计算图
            :param x: 输入
            :param dropout_rate:最后一层使用了dropout
            :return:
            '''
            sess=self.sess
            #fire shape=(输入channel,squeeze channel,expand channel)
            with tf.variable_scope('head'):
                x_conv=self._convlayer(x,'conv1',[3,3,3,64],padding='SAME')                                                 #(N,128,128,3)->(N,128,128,64)
                x_conv=tf.nn.relu(x_conv)
                x_conv=self._convlayer(x_conv,'conv2',[3,3,64,64],padding='SAME')                                           #(N,128,128,64)->(N,128,128,64)
                x_conv=tf.nn.relu(x_conv)
                x_pool1=self._poollayer(x_conv,)

            with tf.variable_scope('fireblock_1'):
                x_fire2=self._fireblock(x_pool1,'fire2',[64,16,32])                                                         # (N,64,64,64)->(N,64,64,64)
                x_fire3=self._fireblock(x_fire2,'fire3',[64,16,32])                                                         # (N,64,64,64)->(N,64,64,64)
                x_fire4=self._fireblock(x_fire3,'fire4',[64,16,64])                                                         # (N,64,64,64)->(N,64,64,128)
                x_pool4=self._poollayer(x_fire4)                                                                            # (N,64,64,128)->(N,32,32,128)

            with tf.variable_scope('fireblock_2'):
                x_fire5=self._fireblock(x_pool4,'fire5',[128,32,64])                                                        # (N,32,32,128)->(N,32,32,128)
                x_fire6=self._fireblock(x_fire5,'fire6',[128,32,64])                                                        # (N,32,32,128)->(N,32,32,128)
                x_fire7=self._fireblock(x_fire6,'fire7',[128,32,192])                                                       # (N,32,32,128)->(N,32,32,384)
                x_pool7=self._poollayer(x_fire7)                                                                            # (N,16,16,384)->(N,16,16,384)

            with tf.variable_scope('fireblock_3'):
                x_fire8=self._fireblock(x_pool7,'fire8',[384,64,192])                                                       # (N,16,16,384)->(N,16,16,384)
                x_fire9=self._fireblock(x_fire8,'fire9',[384,64,192])                                                       # (N,16,16,384)->(N,16,16,384)
                x_fire10=self._fireblock(x_fire9,'fire10',[384,64,256])                                                     # (N,16,16,384)->(N,16,16,512)
                x_pool10=self._poollayer(x_fire10)                                                                          # (N,16,16,512)->(N,8,8,512)

            with tf.variable_scope('tail'):
                x_pool10=tf.nn.dropout(x_pool10,dropout_rate)
                conv_tail=self._convlayer(x_pool10,'conv',[1,1,512,self.outputclass])                                       # (N,8,8,512)->(N,8,8,10)
                conv_tail=tf.nn.relu(conv_tail)
                avg_pool=self._poollayer(conv_tail,pooling='avg',size=(8,8),stride=(1,1),padding='VALID')                   # (N,8,8,10)->(N,1,1,10)
                score=tf.squeeze(avg_pool,[1,2])                                                                            #(N,10)
            #return score
            self.score=score

    def _fireblock(self,x,name,shape):
        '''
        fire模块,先squeeze，在expand
        :param x: 输入
        :param name: 模块名称
        :param shape: （输入X的channel,squeeze 输出channel,expand输出channel）
        :return:
        '''
        #输入X的channel,squeeze 输出channel,expand输出channel
        cur_channel,squ_channel,exp_channel=shape
        #squeeze+activate
        S_1x1=self._convlayer(x,name=name+'/squeeze',shape=[1,1,cur_channel,squ_channel],padding='VALID')
        S_1x1=tf.nn.relu(S_1x1)

        #1x1_expand
        E_1x1=self._convlayer(S_1x1,name=name+'/expand_1x1',shape=[1,1,squ_channel,exp_channel],padding='VALID')
        E_1x1=tf.nn.relu(E_1x1)
        #3x3_expand
        E_3x3=self._convlayer(S_1x1,name=name+'/expand_3x3',shape=[3,3,squ_channel,exp_channel],padding='SAME')
        E_3x3=tf.nn.relu(E_3x3)

        #concat
        x_con=tf.concat([E_1x1,E_3x3],3,name='concat')
        return x_con

    def _convlayer(self,x,name,shape,padding='VALID'):
        '''
        conv函数
        :param x:输入
        :param name: 模块名称
        :param shape: [filter_h,filter_w,input_channel,output_channel,]
        :param padding:
        :return:
        '''
        with tf.variable_scope(name):
            w=tf.get_variable('weight',shape=shape,initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable('biases',shape=shape[-1:],initializer=tf.zeros_initializer())
            conv_=tf.nn.conv2d(x,w,[1,1,1,1],padding=padding)
            conv_=tf.nn.bias_add(conv_,b)
            return conv_

    def _poollayer(self, input, pooling='maxpool',size=(3, 3), stride=(2, 2), padding='SAME'):
        '''
        pool函数
        :param input:输入
        :param pooling: 'avg'表示均值池化，用于最后输出score.'maxpool'表示maxpool
        :param size:
        :param stride:
        :param padding:
        :return:
        '''
        if pooling == 'avg':
            x = tf.nn.avg_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                               padding=padding)
        else:
            x = tf.nn.max_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                           padding=padding)
        return x

    def init_network(self,weight_addr='squeezenet_795.pkl',skip_layer=[]):
        '''
        利用保存好的权值文件来初始化网络
        ！！注意假如需要调用这个函数，一定不可以在调用这个函数后在对全部变量进行初始化 （sess.run(init)）！！
        这样会使得加载的权值被覆盖，要先进行全局变量初始化再读取权值文件！
        :param weight_addr:权值文件
        :return:
        '''
        #初始化全部变量
        sess=self.sess
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

    def save_network_weight(self,filename):
        '''
        提取网络的权值，并保存到文件
        :param filename: 文件名
        :return:
        '''
        sess=self.sess
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

    def predict(self,eyes_images,X_tensor,zero_center=True):
        '''
        前向预测函数，返回计算的概率值
        :param eyes_images:眼部区域的RGB图片
        :param X_tensor:输入X占位符
        :param zero_center:是否需要减去图像均值
        :return:预测的概率
        '''
        cumt_picmean=[103.939, 116.779, 123.68]
        sess=self.sess
        y_score=self.score
        prob=tf.nn.softmax(y_score)
        if zero_center:
            eyes_images=eyes_images-cumt_picmean
        prob_=sess.run(prob,feed_dict={X_tensor:eyes_images})
        return prob_


    def get_eyeimg(self,img,pre_process=True):
        '''
        给定图像，截取图像中的眼部
        下面情况直接返回None：
            1.图像中人脸的数量不为 1
            2.图像中眼睛的数量不为 2
        :param img:
        :return:
        '''
        if self.face_cascade is None:
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        face_cascade=self.face_cascade
        eye_cascade=self.eye_cascade
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3,5)
        if len(faces) !=1:
            print(len(faces))
            print('bad faces')
            return None
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            #检测视频中脸部的眼睛，并用vector保存眼睛的坐标、大小（用矩形表示）
            eyes = eye_cascade.detectMultiScale(roi_gray,scaleFactor=1.2, minNeighbors=7, minSize=(29, 29),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
            #眼睛检测 ,对于识别比较差的情况舍弃
            if len(eyes)!=2:
                print(len(eyes))
                print('bad eyes')
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

        eye_img=roi_color[ey+10:ey+H-2,ex-10:ex+W+10]
        #假如需要，能将图像调整为标准大小
        if pre_process:
            eye_img=cv2.resize(eye_img,(128,128))[:,:,::-1].reshape((1,128,128,3))
        return  eye_img
        


