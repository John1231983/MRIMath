'''
Created on Jul 10, 2018

@author: daniel
'''
#tf.enable_eager_execution()

from DataHandlers.SegNetDataHandler import SegNetDataHandler
import numpy as np
from datetime import datetime
from createSegnetWithIndexPooling import createSegNetWithIndexPooling, createSegNet

from keras.optimizers import SGD, Adam, Adagrad, Adadelta, Nadam
from keras.callbacks import CSVLogger

import sys
import os
from random import choice, sample, shuffle

from CustomLosses import combinedDiceAndChamfer, combinedHausdorffAndDiceMultilabel, combinedHausdorffAndDice, hausdorff_dist, hausdorff_dist_multilabel,dice_coef, chamfer_dist, combinedDiceAndChamferMultilabel, dice_coef_multilabel_loss

from Utils.TimerModule import TimerModule

from CustomLosses import dice_coef_loss, dice_and_iou, dice_coef_multilabel
import shutil
from createSegNetWithIndexPoolingInception import createInceptionSegNet
from Generators.CustomImageAugmentationGenerator import CustomImageAugmentationGenerator
from Generators.CustomImageGenerator import CustomImageGenerator

from keras.utils import np_utils
#from CustomLosses import chamfer_dist
DATA_DIR = os.path.abspath("../")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(DATA_DIR)


     
def main():
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d-%H:%M')
    
    num_training_patients = 30
    num_validation_patients = 3
    
    data_gen = None
    modes = ["flair"]
    dataDirectory = "Data/BRATS_2018/HGG" 
    validationDataDirectory = "Data/BRATS_2018/HGG_Validation"
    testingDataDirectory = "Data/BRATS_2018/HGG_Testing"

    ### Move a random subset of files into validation directory
    if len(os.listdir(validationDataDirectory)) <= 0:
        listOfDirs = os.listdir(dataDirectory)
        shuffle(listOfDirs)
        validation_data = listOfDirs[0:num_validation_patients]
        for datum in validation_data:
            shutil.move(dataDirectory + "/" + datum, validationDataDirectory)
    
    ### Move a random subset of files into testing directory
    if len(os.listdir(testingDataDirectory)) <= 0:
        listOfDirs = os.listdir(dataDirectory)
        shuffle(listOfDirs)
        validation_data = listOfDirs[0:num_validation_patients]
        for datum in validation_data:
            shutil.move(dataDirectory + "/" + datum, testingDataDirectory)
        
        
    dataHandler = SegNetDataHandler("Data/BRATS_2018/HGG", num_patients = num_training_patients, modes = modes)
    dataHandler.setMode("training")
    dataHandler.loadData()
    dataHandler.preprocessForNetwork()
    x_train = dataHandler.X
    x_seg_train = dataHandler.labels
    dataHandler.clear()
    
    dataHandler.setDataDirectory("Data/BRATS_2018/HGG_Validation")
    dataHandler.setNumPatients(num_validation_patients)
    dataHandler.setMode("validation")
    dataHandler.loadData()
    dataHandler.preprocessForNetwork()
    
    x_val = dataHandler.X
    
    x_seg_val = dataHandler.labels
    
    
    dataHandler.clear()
    
    ### Move validation data back to original data directory
    listOfValidationDirs = os.listdir(validationDataDirectory)
    for datum in listOfValidationDirs:
        shutil.move(validationDataDirectory + "/" + datum, dataDirectory)


    input_shape = (dataHandler.W,dataHandler.H, len(modes))
    
    n_labels = 1
    normalize = True
    augmentations = True
    
    if n_labels > 1:
        output_mode = "softmax"
    else:
        output_mode = "sigmoid"

    if augmentations:
        data_gen = CustomImageAugmentationGenerator()
    else:
        data_gen = CustomImageGenerator()

    
    segnet = createInceptionSegNet(input_shape=input_shape, 
                                          n_labels=n_labels, 
                                        output_mode=output_mode)
    
    """
    
    segnet = createSegNet(input_shape, 
                          n_labels = n_labels,
                           kernel=3, 
                           pool_size=(2, 2), 
                           output_mode=output_mode)
    """
    """
    segnet = createSegNetWithIndexPooling(input_shape, 
                                 n_labels, 
                                 32,
                                 output_mode=output_mode, 
                         depth = 1)
    """
    
    num_epochs = 10
    lrate = 1e-3
    adam = Adam(lr = lrate)
    batch_size = 15
    validation_data_gen = CustomImageGenerator()

    if n_labels > 1:
        segnet.compile(optimizer=adam, loss=combinedHausdorffAndDiceMultilabel, metrics=[dice_coef_multilabel])
    else:
        segnet.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])


    model_directory = "Models/segnet_" + date_string 
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    log_info_filename = 'model_loss_log.csv'
    csv_logger = CSVLogger(model_directory + '/' + log_info_filename, append=True, separator=',')
    

    segnet.fit_generator(generator = data_gen.generate(x_train, 
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
    
   

    ## Log everything
    ## Note: should be in a logging class
    model_info_filename = 'model_info.txt'
    model_info_file = open(model_directory + '/' + model_info_filename, "w") 
    model_info_file.write('Number of Patients (training): ' + str(num_training_patients) + '\n')
    model_info_file.write('Number of Patients (validation): ' + str(num_validation_patients) + '\n')

    model_info_file.write('\n\n')
    segnet.summary(print_fn=lambda x: model_info_file.write(x + '\n'))
    model_info_file.close();
    segnet.save(model_directory + '/model.h5')
    
    


    

if __name__ == "__main__":
   main() 
