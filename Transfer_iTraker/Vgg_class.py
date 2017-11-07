'''
VGG_19类文件使用说明：
当做为特征提取器的时候，主要修改的地方是  BuildGraph  函数：（比如我要提取 Conv_Block5中的maxpool_5)
修改：
def BuildGraph(self,images):
    .......
    .......
    self.maxpool_5=maxpool_5
    ......
这样在类的实例中：
vgg_model=Vgg_19(model_path,images,train_list,session,image_batch_size)
conv5_feature=vgg_model.maxpool_5
feature=sess.run(conv5_feature,feed_dict=feed_dict)

就可以用VGG19来提取图片的特征值了
'''
import tensorflow as tf
import numpy as np
VGG_MEAN = [103.939, 116.779, 123.68]
class Vgg_19:
    '''
    VGG类：
    层名称：
    conv1_1->conv1_2->pool1->                           #N,112,112,64
    conv2_1->conv2_2->pool2->                           #N,56,56,128
    conv3_1->conv3_2->conv3_3->conv3_4->pool3->         #N,28,28,256
    conv4_1->conv4_2->conv4_3->conv4_4->pool4->         #N,14,14,512
    conv5_1->conv5_2->conv5_3->conv5_4->pool5->         #N,7,7,512
    fc6->fc7->fc8                                       #4096,4096,1000
    预测：predict->score,prob
    '''
    def __init__(self,model_path,images,train_list,session,image_batch_size):
        '''
        初始化：
        1.载入权值文件，保存
        2.建立计算图
        3.给计算图的变量赋值
        :param model_path: 权值文件地址
        :param images: 输入
        :param train_list: 可以训练的层
        :param session:
        '''
        #载入模型权值文件
        self.modelpath=model_path
        self.image_batch_size=image_batch_size
        self.sess=session
        #初始化建立计算图
        self.BuildGraph(images)
        #将训练好的参数载入到变量中
        print('initing model variable!')
        self.initvariable(train_list)
        print('Vgg19 init Done!')

    def BuildGraph(self,images):
        '''
        建立计算图
        :param images: 输入X
        :return:
        '''
       
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=images)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue -VGG_MEAN[0],
            green -VGG_MEAN[1],
            red -VGG_MEAN[2],
            ])
        #BLOCK 1
        with tf.name_scope('Conv_Block_1'):
            conv1_1=self.conv_layer(bgr,'conv1_1',shape=[3,3,3,64])#224,224,64
            conv1_2=self.conv_layer(conv1_1,'conv1_2',shape=[3,3,64,64])#224,224,64
            maxpool_1=self.maxpool_layer(conv1_2,'pool_1')#112,112,64
        #BLOCK 2
        with tf.name_scope('Conv_Block_2'):
            conv2_1=self.conv_layer(maxpool_1,'conv2_1',shape=[3,3,64,128])#112,112,128
            conv2_2=self.conv_layer(conv2_1,'conv2_2',shape=[3,3,128,128])#112,112,128
            maxpool_2=self.maxpool_layer(conv2_2,'pool_2')#56,56,128
        #BLOCK 3
        with tf.name_scope('Conv_Block_3'):
            conv3_1=self.conv_layer(maxpool_2,'conv3_1',shape=[3,3,128,256])#56,56,256
            conv3_2=self.conv_layer(conv3_1,'conv3_2',shape=[3,3,256,256])#56,56,256
            conv3_3=self.conv_layer(conv3_2,'conv3_3',shape=[3,3,256,256])#56,56,256
            conv3_4=self.conv_layer(conv3_3,'conv3_4',shape=[3,3,256,256])#56,56,256
            maxpool_3=self.maxpool_layer(conv3_4,'pool_3')#28,28,256
        #BLOCK 4
        with tf.name_scope('Conv_Block_4'):
            conv4_1=self.conv_layer(maxpool_3,'conv4_1',shape=[3,3,256,512])#28,28,512
            conv4_2=self.conv_layer(conv4_1,'conv4_2',shape=[3,3,512,512])#28,28,512
            conv4_3=self.conv_layer(conv4_2,'conv4_3',shape=[3,3,512,512])#28,28,512
            conv4_4=self.conv_layer(conv4_3,'conv4_4',shape=[3,3,512,512])#28,28,512
            maxpool_4=self.maxpool_layer(conv4_4,'pool_4')#14,14,512
        #BLOCK 5
        with tf.name_scope('Conv_Block_5'):
            conv5_1=self.conv_layer(maxpool_4,'conv5_1',shape=[3,3,512,512])#14,14,512
            conv5_2=self.conv_layer(conv5_1,'conv5_2',shape=[3,3,512,512])#14,14,512
            conv5_3=self.conv_layer(conv5_2,'conv5_3',shape=[3,3,512,512])#14,14,512
            conv5_4=self.conv_layer(conv5_3,'conv5_4',shape=[3,3,512,512])#14,14,512
            maxpool_5=self.maxpool_layer(conv5_4,'pool_5')#7,7,512
        #BLOCK 6
        with tf.name_scope('FC_Block_6'):
            x_flatten=tf.reshape(maxpool_5,[-1,25088])#7*7*512=25088
            fc6=tf.nn.relu(self.fc_layer(x_flatten,'fc6',shape=[25088,4096]))#N,4096
        #BLOCK 7
        with tf.name_scope('FC_Block_7'):
            fc7=tf.nn.relu(self.fc_layer(fc6,'fc7',shape=[4096,4096]))#N,4096
        #BLOCK 8
        with tf.name_scope('FC_Block_8'):
            score=self.fc_layer(fc7,'fc8',shape=[4096,1000])#N,1000
        self.score=score
        self.pool5=x_flatten
        #return score

    def initvariable(self,train_list):
        '''
        初始化变量
        :param train_list:声明哪些参数可以训练
        :return:
        '''
        layer_name=[]
        sess=self.sess
        #加载VGG模型参数
        model_addr=self.modelpath
        model_dict=np.load(model_addr, encoding='latin1').item()
        #name_list=list(model_dict.keys())
        for n in model_dict:
            layer_name.append(n)
            print('Init layer:{}'.format(n))
            #查看这个变量是否能够训练
            train_flag=True if n in train_list else False
            with tf.variable_scope(n,reuse=True):
                #对 w,b赋值
                var_w=tf.get_variable('weight',trainable=train_flag)
                sess.run(var_w.assign(model_dict[n][0]))
                var_b=tf.get_variable('biases',trainable=train_flag)
                sess.run(var_b.assign(model_dict[n][1]))

        self.layer_name=layer_name
        print('Init Variable Done,model weight has been delete!')

    def conv_layer(self,x,name,shape):
        '''
        卷积层函数
        :param x:输入X
        :param name: 卷积层名称
        :return: 卷积+relu后的结果
        '''
        with tf.variable_scope(name):
            w=tf.get_variable('weight',shape=shape)
            b=tf.get_variable('biases',shape=shape[-1:])
            conv_=tf.nn.conv2d(x,w,[1,1,1,1],'SAME')
            return tf.nn.relu(conv_)
    def maxpool_layer(self,x,name):
        '''
        池化函数
        :param x:输入
        :param name: 层名字
        :return: 池化后的结果
        '''
        with tf.variable_scope(name):
            return tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],'SAME',name='maxpool')
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
    def loadweightfile(self):
        '''
        加载权值文件
        :return:
        '''
        model_addr=self.modelpath
        model_dict=np.load(model_addr, encoding='latin1').item()
        self.model_dict=model_dict
        print('Load VGG19 Model Done!')

    #测试函数，看权值是否正常加载
    def getvar(self,name):
        sess=self.sess
        with tf.variable_scope(name,reuse=True):
            var=tf.get_variable('weight')
            print (sess.run(tf.reduce_mean(var)))
    def variable_check(self):
        sess=self.sess
        ln=self.layer_name
        for n in ln:
            with tf.variable_scope(n,reuse=True):
                var=tf.get_variable('weight')
                print(var.name,sess.run(tf.reduce_mean(var)))