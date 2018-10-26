from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU


def createSegNet(input_shape, n_labels, kernel=3, output_mode="sigmoid"):
    segnet = Sequential()
    img_h  = input_shape[0]
    img_w = input_shape[1]
    encoding_layers = [
        
        
        Convolution2D(32, (kernel, kernel), padding='same'),
        BatchNormalization(),
        PReLU(),
        Convolution2D(32, (kernel, kernel), padding='same'),
        BatchNormalization(),
        PReLU(),
        MaxPooling2D(),
        
        
        Convolution2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        PReLU(),
        Convolution2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        PReLU(),
        MaxPooling2D(),
        

        
        #MaxPooling2D(size=(1,1)),
    ]
    
    segnet.encoding_layers = encoding_layers
    
    for l in segnet.encoding_layers:
        segnet.add(l)
        #print(l.input_shape,l.output_shape,l)
    
    decoding_layers = [
        

        UpSampling2D(),
        Convolution2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        PReLU(),
        Convolution2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        PReLU(),
        
        
        UpSampling2D(),
        Convolution2D(32, (kernel, kernel), padding='same'),
        BatchNormalization(),
        PReLU(),
        Convolution2D(32, (kernel, kernel), padding='same'),
        BatchNormalization(),
        PReLU(),
        Convolution2D(n_labels, (1, 1), padding='valid'),
        BatchNormalization(),

    ]
    segnet.decoding_layers = decoding_layers
    for l in segnet.decoding_layers:
        segnet.add(l)
        #print(l.input_shape,l.output_shape,l)


    segnet.add(Reshape((n_labels, img_h * img_w)))
    segnet.add(Permute((2, 1)))
    segnet.add(Activation(output_mode))
   


    return segnet


                                                                                                                     
                                                                                                                           