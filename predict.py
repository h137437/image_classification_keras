"""
author:tslgithub
email:mymailwith163@163.com
time:2018-12-12
msg: You can choose the following model to train your image, and just switch in config.py:
msg: You can choose the following model to train your image, and just switch in config.py:
    VGG16,VGG19,InceptionV3,Xception,MobileNet,AlexNet,LeNet,ZF_Net,
    ResNet18,ResNet34,ResNet50,ResNet101,ResNet152,mnist_net
    TSL16
"""
from __future__ import print_function
import argparse
#from config import config
import sys,copy,shutil
import cv2
import os,time
from keras.preprocessing.image import img_to_array
import numpy as np

import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True
tf.Session(config=config1)

parser = argparse.ArgumentParser(description='image classifier Example')
parser.add_argument('--model_name', type=str, default='Xception')
parser.add_argument('--train_data_path', type=str, default='dataset/train')
parser.add_argument('--test_data_path', type=str, default='dataset/test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
parser.add_argument('--normal_size', type=int, default=112)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--classNumber', default=4, type=int)
parser.add_argument('--channles', type=int, default=3)
parser.add_argument('--lr', default=0.001)
parser.add_argument('--lr_reduce_patience', type=int, default=5)
parser.add_argument('--data_augmentation', type=bool, default=False)
parser.add_argument('--monitor', type=str, default='val_loss')
parser.add_argument('--cut', default=False)
parser.add_argument('--rat', default=0.1)
parser.add_argument('--className', default='listen')

config = parser.parse_args()

from Build_model import Build_model

class PREDICT(Build_model):
    def __init__(self,config):
        Build_model.__init__(self,config)

        self.className = config.className
        self.test_data_path = os.path.join(config.test_data_path,self.className)

    def classes_id(self):
        with open('train_class_idx.txt','r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        return lines

    def mkdir(self,path):
        if os.path.exists(path):
            return path
        os.mkdir(path)
        return path

    def Predict(self):
        start = time.time()
        model = Build_model(self.config).build_model()
        if os.path.join(os.path.join(self.checkpoints,self.model_name),self.model_name+'.h5'):
            print('weights is loaded')
        else:
            print('weights is not exist')
        model.load_weights(os.path.join(os.path.join(self.checkpoints,self.model_name),self.model_name+'.h5'))



        if(self.channles == 3):
            data_list = list(
                map(lambda x: cv2.resize(cv2.imread(os.path.join(self.test_data_path, x)),
                                         (self.normal_size, self.normal_size)), os.listdir(self.test_data_path)))
        elif(self.channles == 1):
            data_list = list(
                map(lambda x: cv2.resize(cv2.imread(os.path.join(self.test_data_path, x), 0),
                                         (self.normal_size, self.normal_size)), os.listdir(self.test_data_path)))

        i,j,tmp = 0,0,[]
        for img in data_list:
            img = np.array([img_to_array(img)],dtype='float')/255.0
            pred = model.predict(img).tolist()[0]
            label = self.classes_id()[pred.index(max(pred))]
            confidence = max(pred)
            print('predict label     is: ',label)
            print('predict confidect is: ',confidence)

            if label != self.className:
                print('____________________wrong label____________________', label)
                i+=1
            else:
                j+=1

        accuracy = (1.0*j/ (1.0*len(data_list)))*100.0
        print("accuracy:{:.5}%".format(str(accuracy) ))
        print('Done')
        end = time.time()
        print("usg time:",end - start)

        with open("testLog/accuacy.txt","a") as f:
            f.write(config.test_data_path+","+self.className+","+"{:.5}%".format(str(accuracy))+', pic_num: '+str(len(data_list)) + ', acc_num: ' + str(j) + "\n")

def main():
    predict = PREDICT(config)
    predict.Predict()

if __name__=='__main__':
    main()
