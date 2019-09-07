#-*- coding=utf-8 -*-

import numpy as np
import tensorflow as tf

class Model():
    def __init__(self, batch_size, image_height, image_width, ipt_channel, n_classes):
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.ipt_channel = ipt_channel
        self.classes = n_classes  # including background
               
        
    def placeholder(self):
        data_pl = tf.placeholder(tf.float32, shape=(self.batch_size, self.image_height, self.image_width, self.ipt_channel))
        labels_pl   = tf.placeholder(tf.int32, shape=(self.batch_size, ))
        return data_pl, labels_pl
   
    
    def __call__(self, images, keep_prob):     
        flatten1 = self.branch1_random(images)                                                           #shape=(batch, 4096)
        features1 = tf.nn.dropout(flatten1, 0.5)
        logits1 = tf.layers.dense(features1, self.n_classes, name='logits1')

        flatten2 = self.branch2_center(images)                                                           #shape=(batch, 1052)
        features2 = tf.nn.dropout(flatten2, 0.5)
        logits2 = tf.layers.dense(features2, self.n_classes, name='logits2')
        
        outputs = tf.concat([flatten1,flatten2], axis=-1, name='concatenate')
        features = tf.nn.dropout(outputs, 0.5)
        logits = tf.layers.dense(features, self.n_classes, name='logits')
        variables = [var for var in tf.trainable_variables()]
        return logits, outputs, variables
    
    
    def branch1_random(self, images)    
        images = tf.random_crop(images, (self.batch_size, 16, 16, self.ipt_channel))                                    # (batch,16,16,48)
        
        with tf.variable_scope('branch1_random', initializer=tf.truncated_normal_initializer(stddev=0.01)):
            layers1 = tf.layers.conv2d(images, filters = 64, kernel_size = 3, strides = 1, padding = 'same', name = 'layers1')
            layers1 = _instance_norm('BN1',layers1)
            layers1 = tf.nn.leaky_relu(layers1, 0.2, 'activations1')
            pool1 = tf.nn.max_pool(layers1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = 'pool1')   # (batch,8,8,64)
    
            layers2 = tf.layers.conv2d(pool1, filters = 128, kernel_size = 3, strides = 1, padding = 'same',name = 'layers2')
            layers2 = _instance_norm('BN2',layers2)
            layers2 = tf.nn.leaky_relu(layers2, 0.2, 'activations2')
            pool2 = tf.nn.max_pool(layers2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = 'pool2')   # (batch,4,4,128)
            
            layers3 = tf.layers.conv2d(pool2, filters = 256, kernel_size = 3, strides = 1, padding = 'same',name = 'layers3')
            layers3 = _instance_norm('BN3',layers3)
            layers3 = tf.nn.leaky_relu(layers3, 0.2, 'activations3')                                                     # (batch,4,4,256)

            flatten1 = tf.contrib.layers.flatten(layers3)                                                              # (batch,4096)
            assert flatten1.shape.as_list()[-1]==4096
            return flatten1
 
  
    def branch2_center(self, images):
        images = tf.image.central_crop(images, np.random.uniform(0.47,0.7))                                             # (batch,9,9,64)
    
        with tf.variable_scope('branch2_center', initializer=tf.truncated_normal_initializer(stddev=0.01))
            layers1 = tf.layers.conv2d(images, filters = 64, kernel_size = 3, strides = 1, padding = 'same', name = 'layers1')  
            layers1 = _instance_norm('BN1', layers1)
            layers1 = tf.nn.leaky_relu(layer1, 0.2, 'activations1')   
            pool1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = 'pool1')    # (batch,5,5,64)

            layers2 = tf.layers.conv2d(pool1, filters = 128, kernel_size = 3, strides = 1, padding = 'same', name = 'layers2')
            layers2 = _instance_norm('BN2', layers2)
            layers2 = tf.nn.leaky_relu(layers2, 0.2, 'activations2')  
            layers2 = tf.image.resize_images(layers2, (6,6))
            pool2 = tf.nn.avg_pool(layers2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = 'pool2')      # (batch,3,3,128)

            flatten2 = tf.contrib.layers.flatten(pool2)
            assert flatten2.shape.as_list()[-1]==1052
            return flatten2

  
def _weight_variable(name, shape, mean=0):
  """weight_variable generates a weight variable of a given shape."""
  initializer = tf.truncated_normal_initializer(mean=mean,stddev=0.1)
  var = tf.get_variable(name,shape,initializer=initializer, dtype=tf.float32)
  return var


def _bias_variable(name, shape):
  """bias_variable generates a bias variable of a given shape."""
  initializer = tf.truncated_normal_initializer(mean=0,stddev=0.1)
  var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  return var  
  
def _instance_norm(name, inputs):
  # instance_norm      pixel，对HW做归一化，用在风格化迁移；
  # batch_norm         channel，BHW做归一化，对小batchsize效果不好；
  # layer_norm         batch，CHW归一化，主要对RNN作用明显；
  # group_norm         batch  (C//G)*H*W
  # switchable_norm    加权选取
  with tf.variable_scope(name):
    depth = inputs.get_shape()[3]   #dimension(64)      input.shape=(1, 256, 256, 64)
    scale = _weight_variable("scale", [depth], mean=1.0)   #shape=(64,)
    offset = _bias_variable("offset", [depth])   #shape=(64,)
    
    mean, variance = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)  # (1, 1, 1, 64) nomornize in axis=1&2
    inv = tf.rsqrt(variance + 1e-6) #rsqrt(x)=1./sqrt(x)
    normalized = (inputs-mean)*inv
    return scale*normalized + offset  
