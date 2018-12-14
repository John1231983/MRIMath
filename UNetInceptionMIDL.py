'''
Created on Dec 13, 2018

@author: daniel
'''
'''
Created on Nov 3, 2018

@author: daniel
'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datetime import datetime
from UNetFactory.createUNetInception import createUNetInception
from DataHandlers.SegNetDataHandler import SegNetDataHandler

from keras.callbacks import CSVLogger, LearningRateScheduler
from CustomLosses import dice_coef, dice_coef_loss, dice_coef_multilabel, dice_coef_multilabel_loss,dice_coef_bg, dice_coef_net, dice_coef_ed,dice_coef_et
from Generators.CustomImageAugmentationGenerator import CustomImageAugmentationGenerator
from Generators.CustomImageGenerator import CustomImageGenerator
import math
from keras.optimizers import Adam
from Utils.HardwareHandler import HardwareHandler
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
import numpy as np
from sklearn.model_selection import KFold
def step_decay(epoch):
    initial_lrate = 0.1
    #drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.exp(-math.floor((1+epoch)/epochs_drop))
    #lrate = initial_lrate * math.pow(drop,  
    #       math.floor((1+epoch)/epochs_drop))
    return lrate

def main():
    
    hardwareHandler = HardwareHandler()
    numGPUs = hardwareHandler.getAvailableGPUs() 
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d-%H:%M')
    
    num_training_patients = 210
    
    data_gen = None
    modes = ["flair", "t1ce", "t2"]
    
    dataDirectory = "Data/BRATS_2018/HGG" 

        
    dataHandler = SegNetDataHandler(dataDirectory, num_patients = num_training_patients, modes = modes)
    dataHandler.loadData()
    X = dataHandler.X
    Y = dataHandler.labels

    input_shape = (dataHandler.W,dataHandler.H, len(modes))
    
    n_labels = 4
    normalize = True
    augmentations = False
    
    if n_labels > 1:
        output_mode = "softmax"
    else:
        output_mode = "sigmoid"

    if augmentations:
        data_gen = CustomImageAugmentationGenerator()
    else:
        data_gen = CustomImageGenerator()
        
    num_epochs = 100
    #lrate = 1e-2
    adam = Adam()
    batch_size = 128
    
    validation_data_gen = CustomImageGenerator()
    
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    
    splitted_indices=kfold.split(np.array(X), np.array(Y))

    for train, test in splitted_indices:
        x_train = [X[i] for i in train]
        x_test = [X[i] for i in test]
        y_train = [Y[i] for i in train]
        y_test = [Y[i] for i in test]
    
        if numGPUs > 1:
            with tf.device('/cpu:0'):
                unet_to_save = createUNetInception(input_shape, output_mode, n_labels)
            unet = multi_gpu_model(unet_to_save, numGPUs)
        else:
            unet = createUNetInception(input_shape, output_mode, n_labels)
    
            
    
        if n_labels > 1:
            unet.compile(optimizer=adam, loss=dice_coef_multilabel_loss, metrics=[dice_coef_multilabel,dice_coef_bg,dice_coef_net,dice_coef_ed,dice_coef_et])
            if numGPUs > 1:
                unet_to_save.compile(optimizer=adam, loss=dice_coef_multilabel_loss, metrics=[dice_coef_multilabel,dice_coef_bg,dice_coef_net,dice_coef_ed,dice_coef_et])
    
        else:
            unet.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])
            if numGPUs > 1:
                unet_to_save.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])
    
    
    
        model_directory = "Models/unet_" + date_string 
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
            
        log_info_filename = 'model_loss_log.csv'
        csv_logger = CSVLogger(model_directory + '/' + log_info_filename, append=True, separator=',')
        
        lrate_scheduler = LearningRateScheduler(step_decay)
    
        ## Log additional data about model
        ## Note: should be in a logging class
        model_info_filename = 'model_info.txt'
        model_info_file = open(model_directory + '/' + model_info_filename, "w") 
        model_info_file.write('Number of Patients (training): ' + str(num_training_patients) + '\n')
        model_info_file.write('\n\n')
        unet.summary(print_fn=lambda x: model_info_file.write(x + '\n'))
        model_info_file.close();
        
        print("Training on " + str(numGPUs) + " GPUs")
        unet.fit_generator(generator = data_gen.generate(x_train, 
                                                           y_train, 
                                                           batch_size, 
                                                           n_labels,
                                                           normalize), 
                             epochs = num_epochs,
                             steps_per_epoch = len(x_train) / batch_size, 
                             callbacks = [csv_logger, lrate_scheduler], 
                             use_multiprocessing = True, 
                             workers = 4,
                             shuffle=True,
                             validation_steps= len(x_test) / batch_size,
                             validation_data = validation_data_gen.generate(x_test, 
                                                                            y_test, 
                                                                            batch_size, 
                                                                            n_labels, 
                                                                            normalize))
        
        
        if numGPUs > 1:
            unet_to_save.save(model_directory + '/model.h5')
        else:
            unet.save(model_directory + '/model.h5')




    
    


    

if __name__ == "__main__":
   main() 
