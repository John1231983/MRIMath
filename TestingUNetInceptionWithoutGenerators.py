'''
Created on Nov 3, 2018

@author: daniel
'''

import sys
import os
from numpy import genfromtxt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datetime import datetime
import matplotlib.pyplot as plt
from UNetFactory.createUNetInception import createUNetInception
from DataHandlers.SegNetDataHandler import SegNetDataHandler

from keras.callbacks import CSVLogger, LearningRateScheduler
from CustomLosses import dice_coef, dice_coef_loss, dice_coef_multilabel, dice_coef_multilabel_loss
from Generators.CustomImageAugmentationGenerator import CustomImageAugmentationGenerator
from Generators.CustomImageGenerator import CustomImageGenerator
from random import  shuffle
import shutil
import math
from keras.optimizers import Adam

def main():
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d-%H:%M')
    
    num_training_patients = 45
    num_validation_patients = 5
    num_testing_patients = 5
    
    modes = ["flair", "t1ce", "t2", "t1"]
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
    
    ### Move a random subset of files into testing directory
    if len(os.listdir(testingDataDirectory)) <= 0:
        listOfDirs = os.listdir(dataDirectory)
        shuffle(listOfDirs)
        testing_data = listOfDirs[0:num_testing_patients]
        for datum in testing_data:
            shutil.move(dataDirectory + "/" + datum, testingDataDirectory)
        
        
    dataHandler = SegNetDataHandler("Data/BRATS_2018/HGG", num_patients = num_training_patients, modes = modes)
    dataHandler.loadData()
    dataHandler.preprocessForNetwork()
    x_train = dataHandler.X
    x_seg_train = dataHandler.labels
    dataHandler.clear()
    
    dataHandler.setDataDirectory("Data/BRATS_2018/HGG_Validation")
    dataHandler.setNumPatients(num_validation_patients)
    dataHandler.loadData()
    dataHandler.preprocessForNetwork()
    x_val = dataHandler.X
    x_seg_val = dataHandler.labels
    dataHandler.clear()
    """
    ### Move validation data back to original data directory
    listOfValidationDirs = os.listdir(validationDataDirectory)
    for datum in listOfValidationDirs:
        shutil.move(validationDataDirectory + "/" + datum, dataDirectory)
    """

    input_shape = (dataHandler.W,dataHandler.H, len(modes))
    
    n_labels = 1
    
    if n_labels > 1:
        output_mode = "softmax"
    else:
        output_mode = "sigmoid"

    
    unet = createUNetInception(input_shape, output_mode, n_labels)
    
    num_epochs = 500
    lrate = 1e-3
    adam = Adam(lr = lrate)
    batch_size = 20

    if n_labels > 1:
        unet.compile(optimizer=adam, loss=dice_coef_multilabel_loss, metrics=[dice_coef_multilabel])
    else:
        unet.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])


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
    
    unet.fit(x_train, 
             x_seg_train, 
             batch_size = batch_size, 
             epochs = num_epochs,
             callbacks = [csv_logger], 
             validation_data = (x_val, x_seg_val),
             shuffle = True)
    
   

    unet.save(model_directory + '/model.h5')
    
    
    # show results 
    resultPlot = genfromtxt(model_directory + '/' + log_info_filename)
    plt.plot(resultPlot[:,1], label = "Training")
    plt.plot(resultPlot[:,3], label = "Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Coefficient")
    plt.title("Inception U-Net Dice Coefficient")
    plt.legend()
    plt.show()
    
    


    

if __name__ == "__main__":
   main() 
