'''
Created on Jul 10, 2018

@author: daniel
'''
#tf.enable_eager_execution()

from DataHandlers.SegNetDataHandler import SegNetDataHandler
import numpy as np
from datetime import datetime
from createSegnetWithIndexPooling import createSegNetWithIndexPooling

from keras.optimizers import SGD, Adam
from keras.callbacks import CSVLogger

import sys
import os
np.set_printoptions(threshold=np.inf)

from CustomLosses import combinedDiceAndChamfer
from CustomLosses import dice_coef
from CustomLosses import chamfer_dist

from CustomLosses import dice_coef_loss

#from CustomLosses import chamfer_dist
DATA_DIR = os.path.abspath("../")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(DATA_DIR)


     
def main():
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d-%H:%M')
    
    num_training_patients = 150
    num_validation_patients = 15
    modes = ["flair"]
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


    input_shape = (dataHandler.W,dataHandler.H, len(modes))
    
    #n_labels = x_seg_train.shape[2]
    n_labels = 1
    segnet = createSegNetWithIndexPooling(input_shape=input_shape, n_labels=n_labels, depth=3)
    lrate = 0.1
    momentum = 0.9
    num_epochs = 50
    decay = lrate/num_epochs
    adam = Adam(lr = 0.1)
    sgd = SGD(lr=lrate, momentum=momentum, decay=decay, nesterov=True)
    segnet.compile(optimizer=sgd, loss=dice_coef_loss, metrics=[dice_coef])

    model_directory = "Models/segnet_" + date_string 
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    log_info_filename = 'model_loss_log.csv'
    csv_logger = CSVLogger(model_directory + '/' + log_info_filename, append=True, separator=',')
    
    """
    
    image_datagen = ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    #seed = 1,
    rotation_range=3,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=False)
        

    
    
    image_datagen.fit(x_train, augment=True, seed=1)
    batch_size = 50
    segnet.fit_generator(image_datagen.flow(x_train, x_seg_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train) / batch_size,
                    validation_data=(x_val, x_seg_val),
                    epochs=50)
    
    """
    segnet.fit(x_train, x_seg_train,
                epochs=num_epochs,
                batch_size=50,
                shuffle=True,
                validation_data=(x_val, x_seg_val),
                callbacks = [csv_logger]
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
    
    


    

if __name__ == "__main__":
   main() 
