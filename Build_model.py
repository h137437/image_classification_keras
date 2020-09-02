#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author:"tsl"
# email:"mymailwith163@163.com"
# datetime:19-1-17 下午3:07
# software: PyCharm

from __future__ import print_function
import keras
from MODEL import MODEL,ResnetBuilder,SEResNetXt
import sys
import segmentation_models as sm
from keras_efficientnets import EfficientNetB5,EfficientNetB4,EfficientNetB3,EfficientNetB2,EfficientNetB1,EfficientNetB0
from model.mobilenetv3_large import MobileNetV3_Large
from model.mobilenetv3_small import MobileNetV3_Small
from model.nasnet import NASNetMobile, NASNetLarge, NASNetMiddle
from model.shufflenet import ShuffleNet
from model.shufflenetv2 import ShuffleNetV2
sys.setrecursionlimit(10000)

from keras import backend as K
import model.densenet   #取消densenet模型

class Build_model(object):
    def __init__(self,config):
        self.train_data_path = config.train_data_path
        self.checkpoints = config.checkpoints
        self.normal_size = config.normal_size
        self.channles = config.channles
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.classNumber = config.classNumber
        self.model_name = config.model_name
        self.lr = config.lr
        self.config = config
        # self.default_optimizers = config.default_optimizers
        self.data_augmentation = config.data_augmentation
        self.rat = config.rat
        self.cut = config.cut

    def model_confirm(self,choosed_model):
        if choosed_model == 'VGG16':
            model = MODEL(self.config).VGG16()
        elif choosed_model == 'VGG19':
            model = MODEL(self.config).VGG19()
        elif choosed_model == 'AlexNet':
            model = MODEL(self.config).AlexNet()
        elif choosed_model == 'LeNet':
            model = MODEL(self.config).LeNet()
        elif choosed_model == 'ZF_Net':
            model = MODEL(self.config).ZF_Net()
        elif choosed_model == 'ResNet18':
            model = ResnetBuilder().build_resnet18(self.config)
        elif choosed_model == 'ResNet34':
            model = ResnetBuilder().build_resnet34(self.config)
        elif choosed_model == 'ResNet101':
            model = ResnetBuilder().build_resnet101(self.config)
        elif choosed_model == 'ResNet152':
            model = ResnetBuilder().build_resnet152(self.config)
        elif choosed_model =='mnist_net':
            model = MODEL(self.config).mnist_net()
        elif choosed_model == 'TSL16':
            model = MODEL(self.config).TSL16()
        elif choosed_model == 'ResNet50':
            model = keras.applications.ResNet50(include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   input_shape=(self.normal_size,self.normal_size,self.channles),
                                                   pooling='max',
                                                   classes=self.classNumber)
        elif choosed_model == 'InceptionV3':
            model = keras.applications.InceptionV3(include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   input_shape=(self.normal_size,self.normal_size,self.channles),
                                                   pooling='max',
                                                   classes=self.classNumber)

        elif choosed_model == 'Xception':
            model = keras.applications.Xception(include_top=True,
                                                weights=None,
                                                input_tensor=None,
                                                input_shape=(self.normal_size,self.normal_size,self.channles),
                                                pooling='max',
                                                classes=self.classNumber)
        elif choosed_model == 'MobileNet':
            model = keras.applications.MobileNet(include_top=True,
                                                 weights=None,
                                                 input_tensor=None,
                                                 input_shape=(self.normal_size,self.normal_size,self.channles),
                                                 pooling='max',
                                                 classes=self.classNumber)
        elif choosed_model == 'InceptionResNetV2':
            model = keras.applications.InceptionResNetV2(include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   input_shape=(self.normal_size,self.normal_size,self.channles),
                                                   pooling='max',
                                                   classes=self.classNumber)                                         
        elif choosed_model == 'SEResNetXt':
            model = SEResNetXt(self.config).model

        elif choosed_model == 'DenseNet':
            depth = 40
            nb_dense_block = 3
            growth_rate = 12
            nb_filter = 12
            bottleneck = False
            reduction = 0.0
            dropout_rate = 0.0
        
            img_dim = (self.channles, self.normal_size) if K.image_data_format == 'channels_last' else (
                self.normal_size, self.normal_size, self.channles)
        
            model = densenet.DenseNet(img_dim, classNumber=self.classNumber, depth=depth, nb_dense_block=nb_dense_block,
                                      growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate,
                                      bottleneck=bottleneck, reduction=reduction, weights=None)


        elif choosed_model == 'SENet':
            model = sm.Unet('senet154', input_shape=(self.normal_size, self.normal_size, self.channles), 
                           classes=4, activation='softmax',encoder_weights=None)

            #model.summary()

        elif choosed_model == 'EfficientNetB5':
            model = EfficientNetB5(input_shape=(self.normal_size, self.normal_size, self.channles), 
                           classes=4, weights=None)

        elif choosed_model == 'EfficientNetB4':
            model = EfficientNetB4(input_shape=(self.normal_size, self.normal_size, self.channles), 
                           classes=4, weights=None)
        
        elif choosed_model == 'EfficientNetB3':
            model = EfficientNetB3(input_shape=(self.normal_size, self.normal_size, self.channles), 
                           classes=4, weights=None)

        elif choosed_model == 'EfficientNetB2':
            model = EfficientNetB2(input_shape=(self.normal_size, self.normal_size, self.channles), 
                           classes=4, weights=None)

        elif choosed_model == 'EfficientNetB1':
            model = EfficientNetB1(input_shape=(self.normal_size, self.normal_size, self.channles), 
                           classes=4, weights=None)

        elif choosed_model == 'EfficientNetB0':
            model = EfficientNetB0(input_shape=(self.normal_size, self.normal_size, self.channles), 
                           classes=4, weights=None)

        elif choosed_model == 'MobileNetV3_Large':
            model = MobileNetV3_Large(shape=(self.normal_size, self.normal_size, self.channles), 
                           n_class=4).build()
                           
        elif choosed_model == 'MobileNetV3_Small':
            model = MobileNetV3_Small(shape=(self.normal_size, self.normal_size, self.channles), 
                           n_class=4).build()

        elif choosed_model == 'NASNetLarge':
            model = NASNetLarge(input_shape=(self.normal_size, self.normal_size, self.channles), 
                           weights=None, 
                           use_auxiliary_branch=False,
                           classes=4)
        
        elif choosed_model == 'NASNetMobile':
            model = NASNetMobile(input_shape=(self.normal_size, self.normal_size, self.channles), 
                           weights=None, 
                           use_auxiliary_branch=False,
                           classes=4)
        
        elif choosed_model == 'NASNetMiddle':
            model = NASNetMiddle(input_shape=(self.normal_size, self.normal_size, self.channles), 
                           weights=None, 
                           use_auxiliary_branch=False,
                           classes=4)

        elif choosed_model == 'ShuffleNet':
            model = ShuffleNet(input_shape=(self.normal_size, self.normal_size, self.channles), 
                           classes=4)

        elif choosed_model == 'ShuffleNetV2':
            model = ShuffleNetV2(input_shape=(self.normal_size, self.normal_size, self.channles), 
                           classes=4)

        return model

    def model_compile(self,model):
        adam = keras.optimizers.Adam(lr=self.lr)
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])  # compile之后才会更新权重和模型
        return model

    def build_model(self):
        model = self.model_confirm(self.model_name)
        model = self.model_compile(model)
        return model