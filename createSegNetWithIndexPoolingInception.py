'''
Created on Aug 29, 2018

@author: daniel
'''
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation, Reshape, Permute, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras import regularizers
from keras.layers import concatenate, ZeroPadding2D, Flatten
from Mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D




def inceptionModule(inputs, numFilters = 32):
    
    tower_0 = Convolution2D(numFilters, (1,1), padding='same')(inputs)
    tower_0 = BatchNormalization()(tower_0)
    tower_0 = Activation("relu")(tower_0)
    
    tower_1 = Convolution2D(numFilters, (1,1), padding='same')(inputs)
    tower_1 = BatchNormalization()(tower_1)
    tower_1 = Activation("relu")(tower_1)
    tower_1 = Convolution2D(numFilters, (3,3), padding='same')(tower_1)
    tower_1 = BatchNormalization()(tower_1)
    tower_1 = Activation("relu")(tower_1)
    
    tower_2 = Convolution2D(numFilters, (1,1), padding='same')(inputs)
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = Activation("relu")(tower_2)
    tower_2 = Convolution2D(numFilters, (3,3), padding='same')(tower_2)
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = Activation("relu")(tower_2)
    
    tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(inputs)
    tower_3 = Convolution2D(numFilters, (1,1), padding='same')(tower_3)
    tower_3 = BatchNormalization()(tower_3)
    tower_3 = Activation("relu")(tower_3)
    
    inception_module = concatenate([tower_0, tower_1, tower_2, tower_3], axis = 3)
    
    return inception_module

    
def createInceptionSegNet(input_shape, 
                                       n_labels, 
                                       pool_size=(2, 2),
                                        output_mode="sigmoid"):
        # encoder
    inputs = Input(shape=input_shape)
    
    conv_1 = inceptionModule(inputs)    
    conv_2 = inceptionModule(conv_1)    
    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)
    
    conv_3 = inceptionModule(pool_1)
    conv_4 = inceptionModule(conv_3)
    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)
    
    conv_5 = inceptionModule(pool_2)
    conv_6 = inceptionModule(conv_5)
    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_6)


    ## encoding done, decoding start
    
        
    unpool_1 = MaxUnpooling2D(pool_size)([pool_3, mask_3])
    conv_7 = inceptionModule(unpool_1)
    conv_8 = inceptionModule(conv_7)
    
    unpool_2 = MaxUnpooling2D(pool_size)([conv_8, mask_2])
    conv_9 = inceptionModule(unpool_2)
    conv_10 = inceptionModule(conv_9)
    
    
    unpool_3 = MaxUnpooling2D(pool_size)([conv_10, mask_1])
    conv_11 = inceptionModule(unpool_3)
    conv_12 = inceptionModule(conv_11)
    
    conv_13 = Convolution2D(n_labels, (1, 1), padding='valid')(conv_12)
    
    reshape = Reshape((n_labels, input_shape[0] * input_shape[1]))(conv_13)
    permute = Permute((2, 1))(reshape)
    outputs = Activation(output_mode)(permute)
   

    segnet = Model(inputs=inputs, outputs=outputs)
    return segnet
