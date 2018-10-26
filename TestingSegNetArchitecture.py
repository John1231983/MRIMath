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

from CustomLosses import combinedDiceAndChamfer, dice_coef, chamfer_dist, combinedDiceAndChamferMultilabel

from Utils.TimerModule import TimerModule

from CustomLosses import dice_coef_loss, dice_and_iou, dice_coef_multilabel
import shutil
from createSegNetWithIndexPoolingInception import createInceptionSegNet
from CustomImageAugmentationGenerator import CustomImageAugmentationGenerator
from keras.utils import np_utils
#from CustomLosses import chamfer_dist
DATA_DIR = os.path.abspath("../")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(DATA_DIR)


     
def main():
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d-%H:%M')
    
    num_training_patients = 5
    num_validation_patients = 2
    
    data_gen = CustomImageAugmentationGenerator()
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
    timer = TimerModule()
    timer.startTimer()
    dataHandler.loadData()
    dataHandler.preprocessForNetwork()
    timer.stopTimer()
    print("Took about " + str(timer.getElapsedTime()) + " to load the data")
    x_train = dataHandler.X
    x_seg_train = dataHandler.labels
    dataHandler.clear()
    
    dataHandler.setDataDirectory("Data/BRATS_2018/HGG_Validation")
    dataHandler.setNumPatients(num_validation_patients)
    dataHandler.setMode("validation")
    dataHandler.loadData()
    dataHandler.preprocessForNetwork()
    x_val = np.array(dataHandler.X)
    
    x_seg_val = [label.reshape(label.shape[0] * label.shape[1]) for label in dataHandler.labels]
    
    dataHandler.clear()
    
    ### Move validation data back to original data directory
    listOfValidationDirs = os.listdir(validationDataDirectory)
    for datum in listOfValidationDirs:
        shutil.move(validationDataDirectory + "/" + datum, dataDirectory)


    input_shape = (dataHandler.W,dataHandler.H, len(modes))
    
    n_labels = 1
    normalize = True
    
    if n_labels > 1:
        output_mode = "softmax"
        x_seg_val = np.array(x_seg_val)
        x_seg_val = np_utils.to_categorical(x_seg_val)
    else:
        output_mode = "sigmoid"
        for x in x_seg_val:
            x[x > 0.5] = 1
            x[x < 0.5] = 0
        x_seg_val = np.array(x_seg_val)


    
    if normalize:
        mu = np.mean(x_val)
        sigma = np.std(x_val)
        x_val -= mu
        x_val /= sigma

    
    
    
    segnet = createSegNetWithIndexPooling(input_shape=input_shape,
                                          n_labels = n_labels,
                                          k = 16,
                                          depth =1,
                                          output_mode = output_mode)
    
    """
    segnet = createSegNet(input_shape=input_shape, 
                                          n_labels=n_labels, 
                                          output_mode=output_mode)
    
    """
    num_epochs = 100
    lrate = 0.1
    decay = lrate/num_epochs
    adam = Adam(lr = 0.1)
    sgd = SGD(lr = lrate, decay = decay,nesterov=True)
    batch_size = 100

    if n_labels > 1:
        segnet.compile(optimizer=sgd, loss=combinedDiceAndChamferMultilabel, metrics=[dice_coef_multilabel])
    else:
        segnet.compile(optimizer=sgd, loss=dice_coef_loss, metrics=[dice_coef])


    model_directory = "Models/segnet_" + date_string 
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    log_info_filename = 'model_loss_log.csv'
    csv_logger = CSVLogger(model_directory + '/' + log_info_filename, append=True, separator=',')
    
    x_train = np.array(x_train)
    mu = np.mean(x_train)
    sigma = np.std(x_train)
    x_train -= mu
    x_train /= sigma
    
    x_seg_train = [label.reshape(label.shape[0] * label.shape[1]) for label in x_seg_train]
    x_seg_train = np.array(x_seg_train)
    segnet.fit(x_train,
                x_seg_train, 
                batch_size=batch_size,
                epochs = num_epochs,
                validation_data = (x_val, x_seg_val),
                callbacks = [csv_logger],
                shuffle=True)
    """
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
                         validation_data = (x_val, x_seg_val))
    """
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
