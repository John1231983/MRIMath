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
from Exploratory_Stuff.DataHandler import DataHandler
#from keras.callbacks import CSVLogger,ReduceLROnPlateau
#from keras.layers import Conv2D, Activation, MaxPooling2D, Reshape, Dense, Flatten, BatchNormalization, Dropout, LeakyReLU, PReLU,concatenate
#from keras.models import Sequential, Model, Input
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
    
    
    dataHandler = DataHandler("Data/BRATS_2018/HGG", BasicNMFComputer(block_dim=8))
    dataHandler.loadData("flair")
    input_img = Input(shape=(dataHandler.W, dataHandler.H, 1))
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
    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    encoder = Model(input_img, decoded)
    encoder.compile(optimizer='adam', loss='binary_crossentropy')

    #x_train = x_train.astype('float32') / 255.
    
    encoder.fit(x_train, x_seg_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_val, x_seg_val),
                )
    
    decoded_imgs = encoder.predict(x_test)
    
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
    """
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
