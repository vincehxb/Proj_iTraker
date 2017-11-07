'''
2017年10月26日16:44:42

    文件描述：
        基于Inception-v3的利用迁移学习来识别眼球注视方向的文件，这个文件主要有两个主要部分：
            1.利用Inception模型来提取图像特征
            2.利用CNN模型来训练这些提取的特征，这两部分独立


    文件使用说明：

        1.提取特征：
            inception_a=inception.Inceptionv3(input_tensorname='ExpandDims:0',output_tensorname='mixed_10/join:0',\
                                  sess=sess,imgfilepath=r'D:\Proj_DL\Code\Proj_EyeTraker\Proj_iTraker\MIT_iTraker\calimg_file_num',)
            #先加载inception 模型
            inception_a.load_inception_model()
            #再提取特征
            inception_a.extrafeature()

        2.训练CNN模型
            inception_a=inception.Inceptionv3(input_tensorname='ExpandDims:0',output_tensorname='mixed_10/join:0',\
                                  sess=sess,imgfilepath=r'D:\Proj_DL\Code\Proj_EyeTraker\Proj_iTraker\MIT_iTraker\calimg_file_num',)

            inception_a.network_trainner()

    特别说明：
        1.文件中有几个部分的文件名是固定的，假如这些文件名修改了会导致错误，这些文件名前面都用 ！！！注释了，修改是可以先查找这些部分
        主要有：
         （1）face_cascade的xml地址
         (2) 特征文件的命名方式，文件夹名称
        （3）模型文件的保存路径

        2.CNN模型针对的是 pool_3 输入tensor，其维度为 [N,8,8,2048],假如需要调整结构或者是针对其他维度的特征训练，需要修改：feature_cnn 函数

'''



import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import os
import time
import cv2
import pickle

class Inceptionv3(object):
    def __init__(self,input_tensorname,output_tensorname,sess,
                 imgfilepath,
                 inceptionmodel_path='inceptionv3_model/tensorflow_inception_graph.pb'):
        '''

        :param input_tensorname: inception模型中输入节点的名称
        :param output_tensorname: inception模型中输出节点的名称
        :param sess:
        :param imgfilepath: 需要转换成特征的图片文件夹路径
        :param inceptionmodel_path: inception预训练模型的地址
        '''
        self.input_tensorname=input_tensorname
        self.output_tensorname=output_tensorname
        self.sess=sess
        self.inceptionmodel_path=inceptionmodel_path
        self.imgfilepath=imgfilepath
        #！！！！
        self.face_cascade=cv2.CascadeClassifier(r'D:\Proj_DL\Code\Proj_EyeTraker\haarcascade_frontalface_default.xml')

    def load_inception_model(self):
        '''
        加载inception 模型
        :return:
        '''
        MODEL_DIR=self.inceptionmodel_path
        input_tensorname=self.input_tensorname
        output_tensorname=self.output_tensorname
        with gfile.FastGFile(MODEL_DIR,'rb') as f:
            graph_def=tf.GraphDef()
            graph_def.ParseFromString(f.read())
        output_tensor,input_tensor=tf.import_graph_def(graph_def,return_elements=[output_tensorname,input_tensorname])
        self.output_tensor=output_tensor
        self.input_tensor=input_tensor

    def extrafeature(self):
        '''
        将图片送入inception网络，提取变量，并且保存
        :return:
        '''

        y_cordinate,face_matrix=[],[]
        fail_counter,file_index=0,0
        s_time=time.time()
        file_root=self.imgfilepath
        img_names=os.listdir(file_root)
        #打乱图片文件名顺序顺序
        index=np.arange(len(img_names))
        np.random.shuffle(index)
        img_names=list(np.array(img_names)[index])
        #加载计算图相关节点
        input_tensor=self.input_tensor
        output_tensor=self.output_tensor
        sess=self.sess
        if not os.path.exists('Feature_File'):
            print('create feature file')
            os.mkdir('Feature_File')
        for i,f in enumerate(img_names):
            lable_id=int(f[:-4].split('_')[-1])
            img=cv2.imread(os.path.join(file_root,f))
            #脸部检测，只处理检测到有且仅有一张脸的情况
            face_op=self.drew_face_eye(img)
            if face_op == None:
                fail_counter+=1
                continue
            _,f,_=face_op
            #将脸部图像送入inception网络，提取特征，并且保存
            f=f.reshape((1,f.shape[0],f.shape[1],f.shape[2]))
            fea_=sess.run(output_tensor,{input_tensor:f})
            lable_=[0]*16
            lable_[lable_id]=1
            y_cordinate.append(lable_)
            face_matrix.append(fea_.astype('float32'))
            if i%100==0:
                print('pic:{},fail count:{},run time:{}'.format(i,fail_counter,time.time()-s_time))
                # s_time=time.time()
            #每5000条特征保存一次
            if (i%5000==0)and (i !=0):
                file_index+=1
                file_name='feature50k_pool3_'+str(file_index)+r'_.pkl'
                fp=open(os.path.join(r'Feature_File/',file_name),'wb')
                face_matrix=np.array(face_matrix).astype('float32')
                y_cordinate=np.array(y_cordinate).astype('uint8')
                pickle.dump(obj={'feature':face_matrix,'label':y_cordinate},file=fp)
                fp.close()
                y_cordinate=[]
                face_matrix=[]
        #保存剩余的特征，提取特征结束
        file_index+=1
        file_name='feature50k_pool3_'+str(file_index)+r'_.pkl'
        fp=open(os.path.join(r'Feature_File/',file_name),'wb')
        face_matrix=np.array(face_matrix).astype('float32')
        y_cordinate=np.array(y_cordinate).astype('uint8')
        pickle.dump(obj={'feature':face_matrix,'label':y_cordinate},file=fp)
        fp.close()
        print('file save')
    def drew_face_eye(self,img):
        '''
        检测用户当前环境是否能够比较好的识别面部特征
        辨识出脸部以及眼部，标记出来
        :param img:
        :return:
        '''
        face_cascade=self.face_cascade
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3,5)
        #确保画面中只有一个人脸识别出来
        if(len(faces)!=1):
            return None
        for (x,y,w,h) in faces:
            roi_color = img[y:y+h, x:x+w]
            #             #保存脸部图像
            face_mat=roi_color
            eye_mat=[]
        return  (img,face_mat,eye_mat)

    def feature_cnn(self,x_input,traing):
        '''
        自定义的CNN网络，用来分类特征
        注意当前结构只是针对 pool_3特征：[N,8,8,2048]
        使用其他特征需要修改
        :param x_input: 特征输入
        :param traing: BN的是否处于训练的标志位
        :return:
        '''
        print('building cnn network,drop out rate:0.3!')
        drop_rate=0.3
        #conv1 3*3/1 (8,8,2048)->(6,6,1024)
        with tf.variable_scope('conv_1'):
            w=tf.get_variable('weight',shape=[3,3,2048,1024],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable('biases',initializer=tf.constant(0.1,shape=[1024,]))
            conv_=tf.nn.conv2d(x_input,w,[1,1,1,1],'VALID')+b
            conv_=tf.layers.batch_normalization(conv_,training=traing)
            conv_=tf.nn.relu(conv_)
        #conv2 3*3/1 (6,6,2048)->(4,4,512)
        with tf.variable_scope('conv_2'):
            w=tf.get_variable('weight',shape=[3,3,1024,512],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable('biases',initializer=tf.constant(0.1,shape=[512,]))
            conv_=tf.nn.conv2d(conv_,w,[1,1,1,1],'VALID')+b
            conv_=tf.layers.batch_normalization(conv_,training=traing)
            conv_=tf.nn.relu(conv_)
            #conv3 3*3/1 (4,4,2048)->(2,2,256)
        with tf.variable_scope('conv_3'):
            w=tf.get_variable('weight',shape=[3,3,512,256],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable('biases',initializer=tf.constant(0.1,shape=[256,]))
            conv_=tf.nn.conv2d(conv_,w,[1,1,1,1],'VALID')+b
            conv_=tf.layers.batch_normalization(conv_,training=traing)
            conv_=tf.nn.relu(conv_)
        #fc4 (N,8192)->(N,2048)
        with tf.variable_scope('fc_4'):
            x_flatten=tf.reshape(conv_,shape=[-1,2*2*256]) # N*256
            w=tf.get_variable('weight',shape=[2*2*256,1024],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable('biases',initializer=tf.constant(0.1,shape=[1024,]))
            y_l1=tf.matmul(x_flatten,w)+b
            y_l1=tf.layers.batch_normalization(y_l1,training=traing)
            y_l1=tf.nn.relu(y_l1)
        #fc5 (N,2048)->(N,1024)
        with tf.variable_scope('fc_5'):
            y_l1=tf.nn.dropout(y_l1,drop_rate)
            w=tf.get_variable('weight',shape=[1024,512],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable('biases',initializer=tf.constant(0.1,shape=[512,]))
            y_l2=tf.matmul(y_l1,w)+b
            y_l2=tf.layers.batch_normalization(y_l2,training=traing)
            y_l2=tf.nn.relu(y_l2)
        #fc6 (N,1024)->(N,16)
        with tf.variable_scope('fc_6'):
            y_l2=tf.nn.dropout(y_l2,drop_rate)
            w=tf.get_variable('weight',shape=[512,16],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable('biases',initializer=tf.constant(0.1,shape=[16,]))
            y_l3=tf.matmul(y_l2,w)
            y_l3=tf.add(y_l3,b,name='y_score')
        self.y_score=y_l3
        print('building cnn network done!')

    def network_trainner(self):
        '''
        网络训练主函数，调用前确保图片以及转换成了特征
        为了避免出现tensorboard的文件重复，每次调用将会利用时间戳来命名log文件
        :return:
        '''
        sess=self.sess
        #Log文件的时间戳
        timestamp=str(int(time.time()*1e7))
        print('Log file name:{}'.format(timestamp))

        #训练网络用的常规定义
        X=tf.placeholder(tf.float32,[None,8,8,2048])
        Y=tf.placeholder(tf.float32,[None,16])
        TRAINGING=tf.placeholder(tf.bool)
        #
        self.feature_cnn(X,TRAINGING)
        y_socre=self.y_score
        LOSS=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_socre,labels=Y))
        TRAIN=tf.train.AdamOptimizer(1e-3).minimize(LOSS)
        tf.summary.scalar('loss',LOSS)
        #ACC
        ACC_C=tf.equal(tf.arg_max(y_socre,1),tf.arg_max(Y,1))
        ACC=tf.reduce_mean(tf.cast(ACC_C,tf.float32))
        tf.summary.scalar('Accuracy',ACC)
        BN_OPS=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        writer_tr=tf.summary.FileWriter(r'./mylog/'+timestamp+r'/train')
        writer_te=tf.summary.FileWriter(r'./mylog/'+timestamp+r'/test')
        merge=tf.summary.merge_all()
        init=sess.run(tf.global_variables_initializer())

        #！！！！
        d=np.load('Feature_File/fea_validate65k.pkl')
        #！！！！修改验证文件记得修改这里
        te_data,te_label=d['feature'],d['label']
        del d
        tr_data,tr_label=self.ReloadData()
        for i in range(1,1000*10):
            #每 1000 个loop 更换训练集
            if i%1000==0:
                print('change traing data')
                del tr_data
                del tr_label
                tr_data,tr_label=self.ReloadData()
            mask=np.random.choice(range(tr_data.shape[0]),128,replace=False)
            x_,y_=tr_data[mask],tr_label[mask]
            sess.run([TRAIN,BN_OPS],{X:x_,Y:y_,TRAINGING:True })

            if i%10==0:
                loss_,acc_,m_,_=sess.run([LOSS,ACC,merge,BN_OPS],{X:x_,Y:y_,TRAINGING:True})
                writer_tr.add_summary(m_,i)
                print('epoch:{},loss:{},accuracy:{}'.format(i,loss_,acc_))
            if i%20==0:
                mask=np.random.choice(range(te_data.shape[0]),256,replace=False)
                x_,y_=te_data[mask],te_label[mask]
                loss_,acc_,m_,_=sess.run([LOSS,ACC,merge,BN_OPS],{X:x_,Y:y_,TRAINGING:True})
                writer_te.add_summary(m_,i)
                print('--epoch:{},loss:{},accuracy:{}'.format(i,loss_,acc_))

    def ReloadData(self):
        '''
        ！！！！！！！！！！！！！！！！
        每调用一次加载两个训练特征文件夹
        这个函数并不通用，修改了文件夹以及特征文件的名都会导致调用失败
        修改请注意
        :return:
        '''
        #特征文件 1~9 随机抽取 2 个文件读取
        index=np.random.choice(range(1,10),2,replace=False)
        file_name=r'Feature_File/feature50k_pool3_'+str(index[0])+'_.pkl'
        d=np.load(file_name)
        fea_data=d['feature'].reshape((d['feature'].shape[0],d['feature'].shape[2],d['feature'].shape[3],d['feature'].shape[4],))
        fea_label=d['label']
        file_name=r'Feature_File/feature50k_pool3_'+str(index[1])+'_.pkl'
        d=np.load(file_name)
        f_=d['feature'].reshape((d['feature'].shape[0],d['feature'].shape[2],d['feature'].shape[3],d['feature'].shape[4],))
        fea_data=np.concatenate((fea_data,f_),axis=0)
        fea_label=np.concatenate((fea_label,d['label']),axis=0)
        return fea_data,fea_label
    def savemodel(self,modelname,globalstep):
        '''
        模型保存函数
        ！！！
        保存位置为绝对位置，修改需要直接修改函数
        :param modelname: 模型名
        :param globalstep: i
        :return:
        '''
        #保存
        saver=tf.train.Saver()
        saver.save(self.sess,r'./model_save/inception_pool3_smallcnn/'+modelname+'.ckpt'
                   ,global_step=globalstep)
        print('model save!')