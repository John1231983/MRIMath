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
from Exploratory_Stuff.BlockDataHandler import BlockDataHandler
#from keras.callbacks import CSVLogger,ReduceLROnPlateau
from keras.layers import Dense, BatchNormalization, Conv1D, Dropout
from keras.models import Sequential
from keras.callbacks import CSVLogger
#from keras.optimizers import SGD
#import os
from Utils.EmailHandler import EmailHandler
from Utils.HardwareHandler import HardwareHandler
from datetime import datetime
#from keras.utils.training_utils import multi_gpu_model
#import keras.backend as K
#import tensorflow as tf
import nibabel as nib
import numpy as np
from NMFComputer.BasicNMFComputer import BasicNMFComputer
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import PReLU
from Exploratory_Stuff.ExtendedBlockDataHandler import ExtendedBlockDataHandler
 
from NMFComputer.SKNMFComputer import SKNMFComputer
import sys
import os
import cv2
DATA_DIR = os.path.abspath("../")
sys.path.append(DATA_DIR)
from numpy import genfromtxt

def main():
    hardwareHandler = HardwareHandler()
    emailHandler = EmailHandler()
    timer = TimerModule()
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d_%H_%M')
    
    print('Loading the data! This could take some time...')
    num_training_patients = 12;
    num_validation_patients = 1;
    nmfComp = BasicNMFComputer(block_dim=8, num_components=8)
    dataHandler = ExtendedBlockDataHandler("Data/BRATS_2018/HGG", nmfComp, num_patients = num_training_patients, modes = ["flair", "t1ce_bf_corrected", "t1_bf_corrected", "t2_bf_corrected"])
    dataHandler.loadData()
    dataHandler.preprocessForNetwork()
    x_train = dataHandler.X
    labels = dataHandler.labels
    dataHandler.clear()
    
    dataHandler.setLoadingMode("validation")

    dataHandler.setDataDirectory("Data/BRATS_2018/HGG_Validation")
    dataHandler.setNumPatients(num_validation_patients)
    dataHandler.loadData()
    dataHandler.preprocessForNetwork()
    x_val = dataHandler.X
    val_labels = dataHandler.labels
    dataHandler.clear()
    
    print('Building the model now!')
    model = Sequential()
    model.add(Dense(2048, input_dim=len(dataHandler.modes)*dataHandler.nmfComp.num_components))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))
    
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))
    
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))
    
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dense(labels.shape[1], activation='softmax'))
    
    
# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    
    model_directory = "/home/daniel/eclipse-workspace/MRIMath/Models/extended_blocknet_" + date_string
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    log_info_filename = 'model_loss_log.csv'
    csv_logger = CSVLogger(model_directory + '/' + log_info_filename, append=True, separator=',')
    
    model_info_filename = 'model_info.txt'
    model_info_file = open(model_directory + '/' + model_info_filename, "w") 
    model_info_file.write('Number of Patients (training): ' + str(num_training_patients) + '\n')
    model_info_file.write('Number of Patients (validation): ' + str(num_validation_patients) + '\n')

    model_info_file.write('Block Dimensions: ' + str(dataHandler.nmfComp.block_dim) + '\n')
    model_info_file.write('Number of Components (k): ' + str(dataHandler.nmfComp.num_components) + '\n')
    model_info_file.write('\n\n')
    model.summary(print_fn=lambda x: model_info_file.write(x + '\n'))
    model_info_file.close();

    print('Training network!')
    model.fit(x_train,
               labels,
                epochs=500,
                validation_data=(x_val, val_labels),
                callbacks = [csv_logger],
                batch_size=x_train.shape[0])
    
    
    model.save(model_directory + '/model.h5')
    test_data_dir = "Data/BRATS_2018/HGG"
    image = None
    seg_image = None
    m = nmfComp.block_dim
    
    for subdir in os.listdir(test_data_dir):
        seg_est = np.zeros(shape=(dataHandler.W, dataHandler.H))
        data_dirs = os.listdir(test_data_dir + "/" + subdir)
        seg_image = nib.load(test_data_dir + "/" + subdir + "/" +  [s for s in data_dirs if "seg" in s][0]).get_data()
        inds = [i for i in list(range(155)) if np.count_nonzero(seg_image[:,:,i]) > 0]
        for k in inds:
            X_test = []

            foo = []
            for path in data_dirs:
                for mode in dataHandler.modes:
                    if mode in path:
                        image = nib.load(test_data_dir + "/" + subdir + "/" + path).get_data()
                        foo.extend(dataHandler.processData2(image[:,:,k]))
                        
            chunks = [foo[x:x+int(len(foo)/len(dataHandler.modes))] for x in range(0, len(foo), int(len(foo)/len(dataHandler.modes)))]
            for i in range((int(len(foo)/len(dataHandler.modes)))):
                X_test.append(np.concatenate((chunks[0][i], chunks[1][i], chunks[2][i], chunks[3][i]), axis=None))

            print(len(X_test))
            est_labels = [model.predict(x.reshape((1, -1))) for x in X_test]
            #labels = model.predict(H.T)
            ind = 0
            for i in range(0, dataHandler.W, m):
                for j in range(0, dataHandler.H, m):
                    seg_est[i:i+m, j:j+m] = np.full((m, m), np.argmax(est_labels[ind]))
                    ind = ind+1
            fig = plt.figure()
            plt.gray();
    
            a=fig.add_subplot(1,2,1)
            plt.imshow(seg_image[:,:,k])
            plt.axis('off')
            plt.title('GT Segment')
            
            a=fig.add_subplot(1,2,2)
            plt.imshow(seg_est)
            plt.axis('off')
            plt.title('Estimate Segment')
            plt.show()

        
# evaluate the model

if __name__ == "__main__":
   main() 
