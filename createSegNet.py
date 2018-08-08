from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization


def createSegNet(input_shape, n_labels, kernel=3, pool_size=(2, 2), output_mode="softmax"):
    # encoder
    inputs = Input(shape=input_shape)
    segnet = Sequential()
    segnet.add(Conv2D(64, (kernel, kernel), padding="same"))
    segnet.add(BatchNormalization())
    segnet.add(Activation("relu"))
    segnet.add(Conv2D(64, (kernel, kernel), padding="same"))
    segnet.add(BatchNormalization())
    segnet.add(Activation("relu"))
    
    
    segnet.add(UpSampling2D())
    segnet.add(Conv2D(64, (kernel, kernel), padding='same'))
    segnet.add(BatchNormalization())
    segnet.add(Activation('relu'))
    segnet.add(Conv2D(64, (kernel, kernel), padding='same'))
    segnet.add(BatchNormalization())
    segnet.add(Activation('relu'))
    segnet.add(Reshape((1, input_shape[0] * input_shape[1])))
    segnet.add(Permute((2, 1)))
    segnet.add(Activation('softmax'))
   


    return segnet

                                                                                                                           
                                                                                                                           