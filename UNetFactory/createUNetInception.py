'''
Created on Oct 30, 2018

@author: daniel
'''

from keras.models import Model, Input
from keras.layers import Convolution2D, Reshape,Activation, BatchNormalization,MaxPooling2D, Convolution2DTranspose, concatenate

def inceptionModule(inputs, numFilters = 32):
    
    tower_0 = Convolution2D(numFilters, (1,1), padding='same', kernel_initializer = 'he_normal')(inputs)
    tower_0 = BatchNormalization()(tower_0)
    tower_0 = Activation("relu")(tower_0)
    
    tower_1 = Convolution2D(numFilters, (1,1), padding='same',kernel_initializer = 'he_normal')(inputs)
    tower_1 = BatchNormalization()(tower_1)
    tower_1 = Activation("relu")(tower_1)
    tower_1 = Convolution2D(numFilters, (3,3), padding='same',kernel_initializer = 'he_normal')(tower_1)
    tower_1 = BatchNormalization()(tower_1)
    tower_1 = Activation("relu")(tower_1)
    
    tower_2 = Convolution2D(numFilters, (1,1), padding='same',kernel_initializer = 'he_normal')(inputs)
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = Activation("relu")(tower_2)
    tower_2 = Convolution2D(numFilters, (3,3), padding='same',kernel_initializer = 'he_normal')(tower_2)
    tower_2 = Convolution2D(numFilters, (3,3), padding='same',kernel_initializer = 'he_normal')(tower_2)
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = Activation("relu")(tower_2)
    
    tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(inputs)
    tower_3 = Convolution2D(numFilters, (1,1), padding='same',kernel_initializer = 'he_normal')(tower_3)
    tower_3 = BatchNormalization()(tower_3)
    tower_3 = Activation("relu")(tower_3)
    
    inception_module = concatenate([tower_0, tower_1, tower_2, tower_3], axis = 3)
    return inception_module
    
def createUNetInception(input_shape = (240,240,1), output_mode="sigmoid", n_labels = 1):
    inputs = Input(input_shape)
    
    numFilters = 32;

    conv1 = inceptionModule(inputs, numFilters)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = inceptionModule(pool1, 2*numFilters)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = inceptionModule(pool2, 4*numFilters)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = inceptionModule(pool3, 8*numFilters)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = inceptionModule(pool4,16*numFilters)

    up6 = Convolution2DTranspose(8*numFilters, (3,3),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    merge6 = concatenate([conv4,up6],axis=3)
    conv6 = inceptionModule(merge6, 8*numFilters)
    
    up7 = Convolution2DTranspose(4*numFilters,(3,3),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    merge7 =concatenate([conv3,up7],axis=3)
    conv7 = inceptionModule(merge7, 4*numFilters)
    
    up8 = Convolution2DTranspose(2*numFilters,(3,3),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    merge8 =concatenate([conv2,up8],axis=3)
    conv8 = inceptionModule(merge8,2*numFilters)
    
    up9 = Convolution2DTranspose(numFilters,(3,3),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    merge9 =concatenate([conv1,up9],axis=3)
    
    #conv9 = inceptionModule(merge9, numFilters)
    conv9 = Convolution2D(numFilters, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)
    
    conv10 = Convolution2D(n_labels, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = BatchNormalization()(conv10)
    """
    conv10 = Reshape(
             (input_shape[0] * input_shape[1], n_labels),
             input_shape=(input_shape[0], input_shape[1], n_labels))(conv10)
    """
    outputs = Activation(output_mode)(conv10)
    model = Model(input = inputs, output = outputs)
 
    return model


    return model