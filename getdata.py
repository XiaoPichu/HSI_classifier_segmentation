 # python3
#-*-coding:utf-8-*-
import numpy as np
# import keras
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input,Conv2D,BatchNormalization,MaxPool2D, Flatten, Dense,Activation,Dropout,GlobalMaxPool2D,Reshape
import argparse
from random import shuffle
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ModelCheckpoint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import os
from scipy.misc import imread
from scipy.misc import imresize
###################################################  data structure
data_path = "E:\\CZD\\Desktop\\data\\NXP_NET_CZD\\"  #cola1 cola2  0.4 valid
train_txt = "train.txt"
valid_txt = "valid.txt"
num_class = 0  ######including background
def organize_datas():
    classes = os.listdir(data_path)
    traintxt = open(train_txt,'w')
    validtxt = open(valid_txt,'w')
    val_ratio = 0.4
    global num_class
    num_class = len(classes)

    for classid in classes:
        #print("classid",classid)
        imgids = os.listdir(os.path.join(data_path,classid))
        classfile = [os.path.join(data_path,classid,imgid) for imgid in imgids]
        #print(classfile)
        shuffle(classfile)
        for i in range(len(classfile)):
            if i < int(val_ratio*len(classfile)):
                validtxt.write(classfile[i]+'\n')
                validtxt.write(classid+'\n')
            else:
                traintxt.write(classfile[i]+'\n')
                traintxt.write(classid+'\n')             
    traintxt.close()
    validtxt.close()

###################################################  netword structure
def NXP_NET_CZD(net_input_shape,lrate,epochs,global_maxpool):
    inputs = Input(shape=net_input_shape) 
    conv_1 = Conv2D(32, (3,3), padding="same", strides=(1,1), use_bias=False)(inputs)
    batch1 = BatchNormalization()(conv_1)
    batch1 = Activation('relu')(batch1)
    
    conv_2 = Conv2D(32, (3,3), padding="same", strides=(1,1), use_bias=False)(batch1)
    batch2 = BatchNormalization()(conv_2)
    batch2 = Activation('relu')(batch2)
    conv_2 = Conv2D(32, (3,3), padding="same", strides=(1,1), use_bias=False)(batch2)
    batch2 = BatchNormalization()(conv_2)
    batch2 = Activation('relu')(batch2)
    conv_2 = Conv2D(32, (3,3), padding="same", strides=(1,1), use_bias=False)(batch2)
    batch2 = BatchNormalization()(conv_2)
    batch2 = Activation('relu')(batch2)
    
    maxpool_2 = MaxPool2D(pool_size=(2,2))(batch2)
    
    conv_3 = Conv2D(64, (3,3), padding="same", strides=(1,1), use_bias=False)(maxpool_2)
    batch3 = BatchNormalization()(conv_3)
    batch3 = Activation('relu')(batch3)
    conv_3 = Conv2D(64, (3,3), padding="same", strides=(1,1), use_bias=False)(batch3)
    batch3 = BatchNormalization()(conv_3)
    batch3 = Activation('relu')(batch3)
    conv_3 = Conv2D(64, (3,3), padding="same", strides=(1,1), use_bias=False)(batch3)
    batch3 = BatchNormalization()(conv_3)
    batch3 = Activation('relu')(batch3)
    
    maxpool_3 = MaxPool2D(pool_size=(2,2))(batch3)
    
    conv_4 = Conv2D(128, (3,3), padding="same", strides=(1,1), use_bias=False)(maxpool_3)
    batch4 = BatchNormalization()(conv_4)
    batch4 = Activation('relu')(batch4)
    conv_4 = Conv2D(128, (3,3), padding="same", strides=(1,1), use_bias=False)(batch4)
    batch4 = BatchNormalization()(conv_4)
    batch4 = Activation('relu')(batch4)
    conv_4 = Conv2D(128, (3,3), padding="same", strides=(1,1), use_bias=False)(batch4)
    batch4 = BatchNormalization()(conv_4)
    batch4 = Activation('relu')(batch4)
    
    maxpool_4 = MaxPool2D(pool_size=(2,2))(batch4)

    if global_maxpool:
        x = GlobalMaxPool2D()(maxpool_4)

    else:
        x = Flatten()(maxpool_4)

    dense_1 = Dense(128, activation='relu')(x)
    outputs = Dense(num_class, activation='softmax')(dense_1)

    model = Model(inputs=inputs, outputs=outputs)
    sgd = SGD(lr=lrate, momentum=0.9, decay= lrate / epochs, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model    
 
########option####################################################
def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="test.hdf5",
                    help="path to output model")
    args = vars(ap.parse_args())
    return args


class Mygenerator(object):
    def __init__(self,input_shape, batch_size,labelencoder):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.labelencoder = labelencoder
        with open(train_txt, 'r') as f:
            self.train_data = f.readlines()
            self.train_keys = [i for i in range(0,len(self.train_data),2)]
            self.steps_per_epoch = int(len(self.train_keys) / batch_size)
        with open(valid_txt, 'r') as f:
            self.valid_data = f.readlines()
            self.valid_keys = [i for i in range(0,len(self.valid_data),2)]
            self.validation_steps = int(len(self.valid_keys) / batch_size)

    def generator(self, trainflag=True):
        while True:
            if trainflag:
                shuffle(self.train_keys)
                keys = self.train_keys
                datas = self.train_data
                steps = self.steps_per_epoch
            else:
                shuffle(self.valid_keys)
                keys = self.valid_keys
                datas = self.valid_data
                steps = self.validation_steps
            # print(keys)
            for i in range(steps):  # steps多少批
                tmp_inputs = []
                targets = []
                for j in range(self.batch_size):
                    # print('path',datas[keys[j + i * self.batch_size]].strip('\n'))
                    img = imread(datas[keys[j + i * self.batch_size]].strip('\n')).astype('float32')
                    img = imresize(img, self.input_shape).astype('float32')
                    # print(keys[j + i * self.batch_size] + 1)
                    y = int(datas[keys[j + i * self.batch_size] + 1].strip('\n'))
                    y = self.labelencoder.transform([y])
                    y = to_categorical(y, num_class)
                    # print(y)
                    tmp_inputs.append(img)
                    targets.append(y[0])
                final_inputs = np.array(tmp_inputs)
                targets = np.array(targets)
                yield (preprocess_input(final_inputs), targets)


########process###################################################yuyiffge
def trainmodel():
    ########parameters
    net_input_shape = (48,48,3)
    input_shape = (48,48)
    batch_size = 32
    epochs = 3
    lrate = 0.001
    #print('num_class',num_class)
    classes = list(np.arange(num_class)+1)
    print(classes)
    labelencoder = LabelEncoder()
    labelencoder.fit(classes)

    global_maxpool = False
    model = NXP_NET_CZD(net_input_shape,lrate,epochs,global_maxpool)
    modelcheck = ModelCheckpoint(args['model'],monitor='val_acc',save_best_only=True,mode='max')  
    #modelcheck = ModelCheckpoint(args['model'],monitor='val_loss',save_best_only=True,mode='min')  
    callable = [modelcheck]
    
    gen = Mygenerator(input_shape,batch_size,labelencoder)
    print('train steps', gen.steps_per_epoch)
    print('valid steps',gen.validation_steps)
    H = model.fit_generator(gen.generator(True),
                        steps_per_epoch=gen.steps_per_epoch,
                        epochs=epochs,
                        verbose=1,
                        validation_data=gen.generator(False),
                        validation_steps=gen.validation_steps,
                        callbacks=callable)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('./plot.png')  


if __name__=='__main__':
    args = args_parse()
    starttime = time.time()
    organize_datas()
    trainmodel()
    endtime = time.time()
    RUNTIME = endtime - starttime
    print("RUNTIME is:",RUNTIME)
