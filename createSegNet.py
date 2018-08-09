from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization


def createSegNet(input_shape, n_labels, kernel=3, pool_size=(2, 2), output_mode="softmax"):
    img_h  = input_shape[0]
    img_w = input_shape[1]
    encoding_layers = [
        Convolution2D(64, kernel, padding='same', input_shape=( img_h, img_w,1)),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(64, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
    
        Convolution2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
    
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
    
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
    
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        #MaxPooling2D(size=(1,1)),
    ]
    
    segnet = Sequential()
    segnet.encoding_layers = encoding_layers
    
    for l in segnet.encoding_layers:
        segnet.add(l)
        print(l.input_shape,l.output_shape,l)
    
    decoding_layers = [
        UpSampling2D(),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
    
        UpSampling2D(),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
    
        UpSampling2D(),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
    
        UpSampling2D(),
        Convolution2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(n_labels, (1, 1), padding='valid'),
        BatchNormalization(),

    ]
    segnet.decoding_layers = decoding_layers
    for l in segnet.decoding_layers:
        segnet.add(l)
        print(l.input_shape,l.output_shape,l)
    
    segnet.add(Reshape((n_labels, img_h * img_w)))
    segnet.add(Permute((2, 1)))
    segnet.add(Activation('softmax'))
   


    return segnet

                                                                                                                           
                                                                                                                           