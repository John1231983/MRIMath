'''
Created on Oct 30, 2018

@author: daniel
'''

from keras.models import Model, Input
from keras.layers import Convolution2D, Activation, BatchNormalization,MaxPooling2D, Convolution2DTranspose, Dropout, concatenate
from Mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D

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
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = Activation("relu")(tower_2)
    
    tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(inputs)
    tower_3 = Convolution2D(numFilters, (1,1), padding='same',kernel_initializer = 'he_normal')(tower_3)
    tower_3 = BatchNormalization()(tower_3)
    tower_3 = Activation("relu")(tower_3)
    
    inception_module = concatenate([tower_0, tower_1, tower_2, tower_3], axis = 3)
    return inception_module
    
def createUNetInceptionIndexPooling(input_shape = (240,240,1), output_mode="sigmoid", pool_size=(2,2)):
    inputs = Input(input_shape)
    
    numFilters = 16;
    
    conv1 = inceptionModule(inputs, numFilters)
    pool1, mask1 = MaxPoolingWithArgmax2D(pool_size)(conv1)
    
    conv2 = inceptionModule(pool1, 2*numFilters)
    pool2, mask2 = MaxPoolingWithArgmax2D(pool_size)(conv2)
    
    conv3 = inceptionModule(pool2, 4*numFilters)
    pool3, mask3 = MaxPoolingWithArgmax2D(pool_size)(conv3)

    conv4 = inceptionModule(pool3, 8*numFilters)
    drop4 = Dropout(0.5)(conv4)
    pool4, mask4 = MaxPoolingWithArgmax2D(pool_size)(drop4)
    
    up6 = MaxUnpooling2D(pool_size)([pool4, mask4])
    merge6 = concatenate([conv4,up6],axis=3)
    conv6 = inceptionModule(merge6, 4*numFilters)
    
    up7 = MaxUnpooling2D(pool_size)([conv6, mask3])
    merge7 =concatenate([conv3,up7],axis=3)
    conv7 = inceptionModule(merge7, 2*numFilters)
    
    up8 = MaxUnpooling2D(pool_size)([conv7, mask2])
    merge8 =concatenate([conv2,up8],axis=3)
    conv8 = inceptionModule(merge8,numFilters)
    
    up9 = MaxUnpooling2D(pool_size)([conv8, mask1])
    merge9 =concatenate([conv1,up9],axis=3)
    conv9 = inceptionModule(merge9, numFilters)
    
    conv9 = Convolution2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Convolution2D(1, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Convolution2D(1, 1, activation = output_mode)(conv9)

    model = Model(input = inputs, output = conv10)

    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    return model