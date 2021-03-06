'''
Created on Nov 3, 2018

@author: daniel
'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datetime import datetime
from UNetFactory.createUNetInception import createUNetInception
from DataHandlers.UNetDataHandler import UNetDataHandler

from keras.callbacks import CSVLogger, LearningRateScheduler
from CustomLosses import dice_coef, dice_coef_loss, dice_coef_multilabel, dice_coef_multilabel_loss, dice_coef_reg_1, dice_coef_reg_2, dice_coef_reg_3
from Generators.CustomImageAugmentationGenerator import CustomImageAugmentationGenerator
from Generators.CustomImageGenerator2 import CustomImageGenerator2
from random import  shuffle
import shutil
import math
from keras.optimizers import Adam
from Utils.HardwareHandler import HardwareHandler
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model


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
    
    num_training_patients = 200
    num_validation_patients = 0
    n_labels = 1

    data_gen = None
    modes = ["flair", "t1ce", "t2"]
    
    dataDirectory = "Data/BRATS_2018/HGG" 
    validationDataDirectory = "Data/BRATS_2018/HGG_Validation"
    testingDataDirectory = "Data/BRATS_2018/HGG_Testing"
    modelDirectory = "Models"
    
    ## create testing, validation, and model directories
    if not os.path.exists(validationDataDirectory):
        os.makedirs(validationDataDirectory)
    if not os.path.exists(testingDataDirectory):
        os.makedirs(testingDataDirectory)
    if not os.path.exists(modelDirectory):
        os.makedirs(modelDirectory)

    ### Move a random subset of files into validation directory
    if len(os.listdir(validationDataDirectory)) <= 0:
        listOfDirs = os.listdir(dataDirectory)
        shuffle(listOfDirs)
        validation_data = listOfDirs[0:num_validation_patients]
        for datum in validation_data:
            shutil.move(dataDirectory + "/" + datum, validationDataDirectory)
    

        
        
    dataHandler = UNetDataHandler("Data/BRATS_2018/HGG", 
                                  num_patients = num_training_patients, 
                                  modes = modes,
                                  n_labels = n_labels)
    dataHandler.loadData()
    x_train = dataHandler.X
    x_seg_train = dataHandler.labels
    dataHandler.clear()
    
    dataHandler.setDataDirectory("Data/BRATS_2018/HGG_Validation")
    dataHandler.setNumPatients(num_validation_patients)
    dataHandler.loadData()
    x_val = dataHandler.X
    x_seg_val = dataHandler.labels
    dataHandler.clear()

    input_shape = (dataHandler.W,dataHandler.H, len(modes))
    
    normalize = True
    augmentations = False
    
    if n_labels > 1:
        output_mode = "sigmoid"
    else:
        output_mode = "sigmoid"

    if augmentations:
        data_gen = CustomImageAugmentationGenerator()
    else:
        data_gen = CustomImageGenerator2()
        
    num_epochs = 25
    adam = Adam(lr = 0.1)
    batch_size = 64
    
    validation_data_gen = CustomImageGenerator2()
    
    if numGPUs > 1:
        with tf.device('/cpu:0'):
            unet_to_save = createUNetInception(input_shape, output_mode, n_labels)
        unet = multi_gpu_model(unet_to_save, numGPUs)
    else:
        unet = createUNetInception(input_shape, output_mode, n_labels)

        

    if n_labels > 1:
        unet.compile(optimizer=adam, loss=dice_coef_multilabel_loss, metrics=[dice_coef_multilabel])
        if numGPUs > 1:
            unet_to_save.compile(optimizer=adam, loss=dice_coef_multilabel_loss, metrics=[dice_coef_multilabel])

    else:
        unet.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])
        if numGPUs > 1:
            unet_to_save.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])



    model_directory = "Models/unet_" + date_string 
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
        
    log_info_filename = 'model_loss_log.csv'
    csv_logger = CSVLogger(model_directory + '/' + log_info_filename, append=True, separator=',')
    
    ## Log additional data about model
    ## Note: should be in a logging class
    model_info_filename = 'model_info.txt'
    model_info_file = open(model_directory + '/' + model_info_filename, "w") 
    model_info_file.write('Number of Patients (training): ' + str(num_training_patients) + '\n')
    model_info_file.write('Number of Patients (validation): ' + str(num_validation_patients) + '\n')
    model_info_file.write('\n\n')
    unet.summary(print_fn=lambda x: model_info_file.write(x + '\n'))
    model_info_file.close();
    
    print("Training on " + str(numGPUs) + " GPUs")
    unet.fit_generator(generator = data_gen.generate(x_train, 
                                                       x_seg_train, 
                                                       batch_size, 
                                                       n_labels,
                                                       normalize), 
                         epochs = num_epochs,
                         steps_per_epoch = len(x_train) / batch_size, 
                         callbacks = [csv_logger], 
                         use_multiprocessing = True, 
                         workers = 4,
                         shuffle=True,
                         validation_steps= len(x_val) / batch_size,
                         validation_data = validation_data_gen.generate(x_val, 
                                                                        x_seg_val, 
                                                                        batch_size, 
                                                                        n_labels, 
                                                                        normalize))
    
    
    if numGPUs > 1:
        unet_to_save.save(model_directory + '/model.h5')
    else:
        unet.save(model_directory + '/model.h5')

    
    


    

if __name__ == "__main__":
   main() 
