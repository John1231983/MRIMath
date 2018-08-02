'''
Created on Jul 10, 2018

@author: daniel
'''

#from multiprocessing import Process, Manager
#from keras.utils import np_utils
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.TimerModule import TimerModule
from Exploratory_Stuff.SegNetDataHandler import SegNetDataHandler
#from keras.callbacks import CSVLogger,ReduceLROnPlateau
from keras.layers import Conv2D, Activation, MaxPooling2D, Reshape, Dense, Flatten, BatchNormalization, Dropout, LeakyReLU, PReLU,concatenate
from keras.models import Sequential, Model, Input
#from keras.optimizers import SGD
#import os
from Utils.EmailHandler import EmailHandler
from Utils.HardwareHandler import HardwareHandler
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
#from keras.utils.training_utils import multi_gpu_model
#import keras.backend as K
#import tensorflow as tf
from Mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D

from NMFComputer.BasicNMFComputer import BasicNMFComputer

from NMFComputer.ProbabilisticNMFComputer import ProbabilisticNMFComputer
import sys
import os
DATA_DIR = os.path.abspath("../")
sys.path.append(DATA_DIR)

def main():
    hardwareHandler = HardwareHandler()
    emailHandler = EmailHandler()
    timer = TimerModule()
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d_%H_%M')
    
    
    dataHandler = SegNetDataHandler("Data/BRATS_2018/HGG", ProbabilisticNMFComputer(block_dim=10), num_patients = 1)
    dataHandler.loadData("flair")
    dataHandler.preprocessForNetwork()
    x_train = dataHandler.X
    x_seg_train = dataHandler.labels
    dataHandler.clear()

    dataHandler.setDataDirectory("Data/BRATS_2018/HGG_Validation")
    dataHandler.loadData("flair")
    dataHandler.preprocessForNetwork()
    x_val = dataHandler.X
    x_seg_val = dataHandler.labels
    dataHandler.clear()
    
    dataHandler.setDataDirectory("Data/BRATS_2018/HGG_Testing")
    dataHandler.loadData("flair")
    dataHandler.preprocessForNetwork()
    x_test = dataHandler.X
    x_seg_test = dataHandler.labels
    dataHandler.clear()

    
    input_shape = (dataHandler.W, dataHandler.H, 1)
    inputs = Input(shape=input_shape)
    kernel = 3
    pool_size=(2, 2)
    output_mode="softmax"
    n_labels = 2
    
    
    conv_1 = Conv2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Conv2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Conv2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Conv2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Conv2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Conv2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Conv2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Conv2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Conv2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Conv2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Conv2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Conv2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Conv2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build enceder done..")

    # decoder

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = Conv2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Conv2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Conv2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = Conv2D(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Conv2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Conv2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Conv2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Conv2D(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Conv2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Conv2D(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Conv2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Conv2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Conv2D(n_labels, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Reshape((input_shape[0] * input_shape[1], n_labels), input_shape=(input_shape[0], input_shape[1], n_labels))(conv_26)

    outputs = Activation(output_mode)(conv_26)
    print("Build decoder done..")

    segnet = Model(inputs=inputs, outputs=outputs, name="SegNet")
    segnet.compile(optimizer='adadelta', loss='categorical_crossentropy')

    
    segnet.fit(x_train, x_seg_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_val, x_seg_val),
                )
    
    decoded_imgs = segnet.predict(x_test)
    
    n = 10
    for i in range(n):
        fig = plt.figure()
        plt.gray();   
            
        a=fig.add_subplot(1,2,1)
        plt.imshow(x_seg_test[i].reshape(dataHandler.W, dataHandler.W))
        plt.axis('off')
        plt.title('Original')
        
        a=fig.add_subplot(1,2,2)
        plt.imshow(decoded_imgs[i].reshape(dataHandler.W, dataHandler.W))
        plt.gray()
        plt.show()
    
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i].reshape(dataHandler.W, dataHandler.W))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(dataHandler.W, dataHandler.W))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
    model = Sequential()
    model.add(Dense(100, input_shape = (310,)))
    model.add(PReLU())
    model.add(Dense(250))
    model.add(PReLU())
    model.add(Dense(5, activation='softmax'))
    
    

    """
    
    num_epochs = 30  
    batchSize = 128 
    lrate = 0.1e-3
    momentum = 0.9
    decay_rate = lrate / num_epochs
    
    training = dataHandler.X
    training_labels = dataHandler.labels
    
    #decay = lrate/num_epochs   
    sgd = SGD(lr=lrate, momentum=momentum, decay=decay_rate, nesterov=True)
    
    # Declaring the two callbacks I use - still having issues using the model checkpoint one
    # Also considering using the Early Termination...
    # csv_logger = CSVLogger(model_directory + '/' + log_info_filename, append=True, separator=',')
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.1e-9)
    
    # Grab the number of available GPUS on the device you're running on - in the case of the HPC, it's 4
    # And on your local lap top, probably 1
    # Based on the result, your model may be converted to a multiple GPU model
    G = hardwareHandler.getAvailableGPUs()
    print('Using ' + str(G) + ' GPUs to train the network!')
    if G > 1:
        parallel_model = multi_gpu_model(model, G)
        parallel_model.compile(optimizer=sgd, loss='binary_crossentropy',metrics = ['accuracy'])
        timer.startTimer()   # this is for timing/profiling purposes    
        parallel_model.fit(training, training_labels,
            epochs=num_epochs,
            batch_size=batchSize * G,
            shuffle=True)
            #validation_data=(testing, testing_labels),
            #callbacks=[csv_logger, reduce_lr])
        timer.stopTimer()
        model.set_weights(parallel_model.get_weights())
    
    else:
        model.compile(optimizer=sgd, loss='binary_crossentropy',metrics = ['accuracy'])
        timer.startTimer()
        model.fit(training, training_labels,
            epochs=num_epochs,
            batch_size=batchSize * G,
            shuffle=True)
            #validation_data=(testing, testing_labels),
            #callbacks=[csv_logger, reduce_lr])
        timer.stopTimer()
"""  
if __name__ == "__main__":
   main() 
