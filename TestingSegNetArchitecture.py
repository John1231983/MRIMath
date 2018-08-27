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
from createSegNet import createSegNet


#from keras.utils.training_utils import multi_gpu_model
#import keras.backend as K
#import tensorflow as tf
from Mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from keras.optimizers import SGD
from keras.callbacks import CSVLogger

from NMFComputer.BasicNMFComputer import BasicNMFComputer

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
    
    num_training_patients = 100
    num_validation_patients = 5
    mode = "flair"
    
    dataHandler = SegNetDataHandler("Data/BRATS_2018/HGG", BasicNMFComputer(block_dim=8), num_patients = num_training_patients)
    dataHandler.loadData("flair")
    dataHandler.preprocessForNetwork()
    x_train = dataHandler.X
    x_seg_train = dataHandler.labels
    dataHandler.clear()
    
    dataHandler.setDataDirectory("Data/BRATS_2018/HGG_Validation")
    dataHandler.setNumPatients(num_validation_patients)
    dataHandler.loadData(mode)
    dataHandler.preprocessForNetwork()
    x_val = dataHandler.X
    x_seg_val = dataHandler.labels
    dataHandler.clear()
    
    dataHandler.setDataDirectory("Data/BRATS_2018/HGG_Testing")
    dataHandler.setNumPatients(1)
    dataHandler.loadData("flair")
    dataHandler.preprocessForNetwork()
    x_test = dataHandler.X
    x_seg_test = dataHandler.labels
    dataHandler.clear()
    

    input_shape = (dataHandler.W,dataHandler.H, 1)
    
    n_labels = x_seg_train.shape[2]
    segnet = createSegNet(input_shape=input_shape, n_labels=n_labels)
    lrate = 0.1
    momentum = 0.9
    #decay = lrate/num_epochs   
    sgd = SGD(lr=lrate, momentum=momentum, nesterov=True)
    segnet.compile(optimizer=sgd, loss='categorical_crossentropy')

    model_directory = "/home/daniel/eclipse-workspace/MRIMath/Models/segnet_" + date_string + "_" + mode
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    log_info_filename = 'model_loss_log.csv'
    csv_logger = CSVLogger(model_directory + '/' + log_info_filename, append=True, separator=',')
    
    segnet.fit(x_train, x_seg_train,
                epochs=10,
                batch_size=30,
                shuffle=True,
                validation_data=(x_val, x_seg_val),
                callbacks = [csv_logger],
                )
    
    
    model_info_filename = 'model_info.txt'
    model_info_file = open(model_directory + '/' + model_info_filename, "w") 
    model_info_file.write('Number of Patients (training): ' + str(num_training_patients) + '\n')
    model_info_file.write('Number of Patients (validation): ' + str(num_validation_patients) + '\n')

    #model_info_file.write('Block Dimensions: ' + str(dataHandler.nmfComp.block_dim) + '\n')
    #model_info_file.write('Number of Components (k): ' + str(dataHandler.nmfComp.num_components) + '\n')
    model_info_file.write('\n\n')
    segnet.summary(print_fn=lambda x: model_info_file.write(x + '\n'))
    model_info_file.close();
    segnet.save(model_directory + '/model.h5')
    
    decoded_imgs = segnet.predict(x_test)
    
    n = 100
    for i in range(n):
        fig = plt.figure()
        plt.gray();   
        a=fig.add_subplot(1,3,1)
        plt.imshow(255*x_test[i,:,:,0])
        plt.axis('off')
        plt.title('Original')
        
        a=fig.add_subplot(1,3,2)
        plt.imshow(np.argmax(x_seg_test[i], axis=-1).reshape(dataHandler.W, dataHandler.W))
        plt.axis('off')
        plt.title('GT Segment')
        
        a=fig.add_subplot(1,3,3)
        plt.imshow(np.argmax(decoded_imgs[i], axis=-1).reshape(dataHandler.W, dataHandler.W))
        plt.axis('off')
        plt.title('Predicted Segment')

        plt.show()
    
    

if __name__ == "__main__":
   main() 
