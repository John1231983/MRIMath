'''
Created on Aug 29, 2018

@author: daniel
'''
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras import regularizers

from Mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from rdflib.plugins.sparql.parser import Drop

import keras.backend


def createSegNetWithIndexPooling(input_shape, n_labels, kernel=3, pool_size=(2, 2), output_mode="sigmoid", depth = 2):
    if depth == 2:
        return create2LayerSegNetWithIndexPooling(input_shape, n_labels, kernel, pool_size, output_mode)
    elif depth == 3:
        return create3LayerSegNetWithIndexPooling(input_shape, n_labels, kernel, pool_size, output_mode)


def create3LayerSegNetWithIndexPooling(input_shape, n_labels, kernel=3, pool_size=(2, 2), output_mode="sigmoid"):
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = PReLU()(conv_1)
    conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
    #conv_2 = Dropout(0.5)(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = PReLU()(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)
    
    conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = PReLU()(conv_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    #conv_4 = Dropout(0.5)(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = PReLU()(conv_4)
    
    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)
    
    conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = PReLU()(conv_5)
    conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
    #conv_6 = Dropout(0.5)(conv_6)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = PReLU()(conv_6)
    
    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_6)
    
    unpool_1 = MaxUnpooling2D(pool_size)([pool_3, mask_3])
    
    conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_1)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = PReLU()(conv_7)
    conv_8 = Convolution2D(256, (kernel, kernel), padding="same")(conv_7)
    #conv_8 = Dropout(0.5)(conv_8)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = PReLU()(conv_8)
    
    unpool_2 = MaxUnpooling2D(pool_size)([pool_2, mask_2])

    conv_9 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_2)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = PReLU()(conv_9)
    conv_10 = Convolution2D(128, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = PReLU()(conv_10)
    
    unpool_3 = MaxUnpooling2D(pool_size)([pool_1, mask_1])
    
    conv_11 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_3)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = PReLU()(conv_11)
    conv_12 = Convolution2D(64, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = PReLU()(conv_12)
  
    conv_13 = Convolution2D(n_labels, (1, 1), padding='valid')(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    
    reshape = Reshape((n_labels, input_shape[0] * input_shape[1]))(conv_13)
    permute = Permute((2, 1))(reshape)
    outputs = Activation(output_mode)(permute)
    
    segnet = Model(inputs=inputs, outputs=outputs)
    return segnet
    
def create2LayerSegNetWithIndexPooling(input_shape, n_labels, kernel=3, pool_size=(2, 2), output_mode="sigmoid"):
        # encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(32, (kernel, kernel), padding="same")(inputs)
    #conv_1 = Dropout(0.5)(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = PReLU()(conv_1)
    
    conv_2 = Convolution2D(32, (kernel, kernel), padding="same")(conv_1)
    #conv_2 = Dropout(0.5)(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = PReLU()(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)
    
    conv_3 = Convolution2D(64, (kernel, kernel), padding="same")(pool_1)
    #conv_3 = Dropout(0.5)(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = PReLU()(conv_3)

    conv_4 = Convolution2D(64, (kernel, kernel), padding="same")(conv_3)
    #conv_4 = Dropout(0.5)(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = PReLU()(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)
         
    unpool_2 = MaxUnpooling2D(pool_size)([pool_2, mask_2])

    conv_9 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_2)
    #conv_9 = Dropout(0.5)(conv_9)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = PReLU()(conv_9)
    conv_10 = Convolution2D(64, (kernel, kernel), padding="same")(conv_9)
    #conv_10 = Dropout(0.5)(conv_10)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = PReLU()(conv_10)
    
    unpool_3 = MaxUnpooling2D(pool_size)([pool_1, mask_1])
    
    conv_11 = Convolution2D(32, (kernel, kernel), padding="same")(unpool_3)
    #conv_11 = Dropout(0.5)(conv_11)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = PReLU()(conv_11)
    conv_12 = Convolution2D(32, (kernel, kernel), padding="same",kernel_regularizer=regularizers.l2(0.05))(conv_11)
    #conv_12 = Dropout(0.5)(conv_12)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = PReLU()(conv_12)
  
    conv_13 = Convolution2D(n_labels, (1, 1), padding='valid')(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    
    reshape = Reshape((n_labels, input_shape[0] * input_shape[1]))(conv_13)
    permute = Permute((2, 1))(reshape)
    outputs = Activation(output_mode)(permute)
   

    segnet = Model(inputs=inputs, outputs=outputs)
    return segnet
