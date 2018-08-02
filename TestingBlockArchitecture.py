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
from keras.layers import Dense
from keras.models import Sequential
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
    
    
    nmfComp = ProbabilisticNMFComputer(block_dim=8)
    dataHandler = BlockDataHandler("Data/BRATS_2018/HGG", nmfComp, num_patients = 5)
    dataHandler.loadData("flair")
    dataHandler.preprocessForNetwork()
    x_train = dataHandler.X
    labels = dataHandler.labels
    dataHandler.clear()
    
    dataHandler.setDataDirectory("Data/BRATS_2018/HGG_Validation")
    dataHandler.setNumPatients(1)
    dataHandler.loadData("flair")
    dataHandler.preprocessForNetwork()
    x_val = dataHandler.X
    val_labels = dataHandler.labels
    dataHandler.clear()
    
        
    model = Sequential()
    model.add(Dense(200, input_dim=dataHandler.nmfComp.num_components, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(labels.shape[1], activation='softmax'))
# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the mode
    model.fit(x_train,
               labels,
                epochs=10,
                validation_data=(x_val, val_labels),
                batch_size=10)
    
    test_data_dir = "Data/BRATS_2018/HGG_Testing"
    mode = "flair"
    image = None
    seg_image = None
    for subdir in os.listdir(test_data_dir):
        for path in os.listdir(test_data_dir+ "/" + subdir):
            if mode in path:
                image = nib.load(test_data_dir + "/" + subdir + "/" + path).get_data()
                seg_image = nib.load(test_data_dir+ "/" + subdir + "/" + path.replace(mode, "seg")).get_data()
                break
    
    m = nmfComp.block_dim
    for k in range(0,155):
        seg_est = np.zeros((dataHandler.W, dataHandler.H))
        _, H = nmfComp.run(image[:,:,k])
        H_cols = np.hsplit(H, H.shape[1])
        labels = [model.predict(x) for x in H_cols]
        ind = 0
        for i in range(0, dataHandler.W, m):
            for j in range(0, dataHandler.H, m):
                seg_est[i:i+m, j:j+m] = labels[ind]
                ind = ind+1
        
        fig = plt.figure()
        plt.gray();
        a=fig.add_subplot(1,2,1)
        plt.imshow(seg_image[:,:,i])
        plt.axis('off')
        plt.title('Original')
        
        a=fig.add_subplot(1,2,2)
        plt.imshow(seg_est)
        plt.gray()
        plt.show()
        
        
# evaluate the model

if __name__ == "__main__":
   main() 
