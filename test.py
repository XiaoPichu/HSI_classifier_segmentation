# -*- coding:utf-8 -*-

import numpy as np
import os
import argparse
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import time
from keras.applications.imagenet_utils import preprocess_input
from scipy.misc import imread
from scipy.misc import imresize
import argparse

net_input_shape = (48,48)
confi = 0.9

path = "E:\\CZD\\Desktop\\test"
imgpaths = os.listdir(path)
batch_size = len(imgpaths)

def Predict(args):
    model = load_model(args.model)
    inputs = []
    for i in range(batch_size):
        print(imgpaths[i])
        img = imread(os.path.join(path,imgpaths[i])).astype('float32')
        img = imresize(img, net_input_shape).astype('float32')
        inputs.append(preprocess_input(img))
    inputs = np.array(inputs)
    preds = model.predict(inputs, batch_size=batch_size, verbose=1)
    print(preds)
    results = np.argmax(preds, axis=1)+1
    print(results)

if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("-model",required=True)
    args = parse.parse_args()
    starttime = time.time()
    Predict(args)
    endtime = time.time()
    runtime = endtime - starttime
    print("runtime is:",runtime)
