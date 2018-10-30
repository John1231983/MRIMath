'''
Created on Aug 29, 2018

@author: daniel
'''
from DataHandlers.SegNetDataHandler import SegNetDataHandler
'''
Created on Jul 10, 2018

@author: daniel
'''

#from multiprocessing import Process, Manager
#from keras.utils import np_utils
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from DataHandlers.UNetDataHandler import UNetDataHandler
from datetime import datetime
import matplotlib.pyplot as plt
from createUNet import createUNet
from keras.optimizers import SGD
from keras.callbacks import CSVLogger
from CustomLosses import combinedDiceAndChamfer, combinedHausdorffAndDice
from CustomLosses import dice_coef, dice_coef_loss, dice_coef_multilabel_loss, dice_coef_multilabel



'''
Created on Aug 29, 2018

@author: daniel
'''
from DataHandlers.SegNetDataHandler import SegNetDataHandler
'''
Created on Jul 10, 2018

@author: daniel
'''

#from multiprocessing import Process, Manager
#from keras.utils import np_utils
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from DataHandlers.UNetDataHandler import UNetDataHandler
from datetime import datetime
import matplotlib.pyplot as plt
from createUNet import createUNet
from keras.optimizers import SGD
from keras.callbacks import CSVLogger
from CustomLosses import combinedDiceAndChamfer, combinedHausdorffAndDice
from CustomLosses import dice_coef
from Generators.CustomImageAugmentationGenerator import CustomImageAugmentationGenerator
from Generators.CustomImageGenerator import CustomImageGenerator
from random import choice, sample, shuffle
import shutil

from keras.optimizers import SGD, Adam, Adagrad, Adadelta, Nadam

def main():
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d-%H:%M')
    
    num_training_patients = 1
    num_validation_patients = 1
    
    data_gen = None
    modes = ["flair", "t1ce", "t2", "t1"]
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
    #dataHandler.preprocessForNetwork()
    x_train = dataHandler.X
    x_seg_train = dataHandler.labels
    dataHandler.clear()
    
    dataHandler.setDataDirectory("Data/BRATS_2018/HGG_Validation")
    dataHandler.setNumPatients(num_validation_patients)
    dataHandler.setMode("validation")
    dataHandler.loadData()
    #dataHandler.preprocessForNetwork()
    
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

    
    unet = createUNet(input_shape, output_mode)
   
    
    num_epochs = 60
    lrate = 1e-3
    adam = Adam(lr = lrate)
    batch_size = 20
    validation_data_gen = CustomImageGenerator()

    if n_labels > 1:
        unet.compile(optimizer=adam, loss=dice_coef_multilabel_loss, metrics=[dice_coef_multilabel])
    else:
        unet.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])


    model_directory = "Models/unet_" + date_string 
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    log_info_filename = 'model_loss_log.csv'
    csv_logger = CSVLogger(model_directory + '/' + log_info_filename, append=True, separator=',')
    

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
    
   

    ## Log everything
    ## Note: should be in a logging class
    model_info_filename = 'model_info.txt'
    model_info_file = open(model_directory + '/' + model_info_filename, "w") 
    model_info_file.write('Number of Patients (training): ' + str(num_training_patients) + '\n')
    model_info_file.write('Number of Patients (validation): ' + str(num_validation_patients) + '\n')

    model_info_file.write('\n\n')
    unet.summary(print_fn=lambda x: model_info_file.write(x + '\n'))
    model_info_file.close();
    unet.save(model_directory + '/model.h5')
    
    


    

if __name__ == "__main__":
   main() 
