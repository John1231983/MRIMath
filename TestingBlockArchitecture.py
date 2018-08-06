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
from keras.layers.advanced_activations import LeakyReLU, PReLU

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
    
    print('Loading the data! This could take some time...')
    mode = "t1ce"
    nmfComp = ProbabilisticNMFComputer(block_dim=8, num_components=8)
    dataHandler = BlockDataHandler("Data/BRATS_2018/HGG", nmfComp, num_patients = 15)
    dataHandler.loadData(mode)
    dataHandler.preprocessForNetwork()
    x_train = dataHandler.X
    labels = dataHandler.labels
    dataHandler.clear()
    
    dataHandler.setDataDirectory("Data/BRATS_2018/HGG_Validation")
    dataHandler.setNumPatients(2)
    dataHandler.loadData(mode)
    dataHandler.preprocessForNetwork()
    x_val = dataHandler.X
    val_labels = dataHandler.labels
    dataHandler.clear()
    
    print('Building the model now!')
    model = Sequential()
    model.add(Dense(1000, input_dim=dataHandler.nmfComp.num_components))
    model.add(LeakyReLU())
    model.add(Dense(500))
    model.add(LeakyReLU())
    model.add(Dense(250))
    model.add(LeakyReLU())
    model.add(Dense(100))
    model.add(LeakyReLU())
    model.add(Dense(labels.shape[1], activation='softmax'))
    
    
# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    
    model_directory = "/home/daniel/eclipse-workspace/MRIMath/Models/" + date_string + "_" + mode
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    log_info_filename = 'model_loss_log.csv'
    csv_logger = CSVLogger(model_directory + '/' + log_info_filename, append=True, separator=',')
    
    print('Training network!')
    model.fit(x_train,
               labels,
                epochs=200,
                validation_data=(x_val, val_labels),
                callbacks = [csv_logger],
                batch_size=x_train.shape[0])
    
    
    model.save(model_directory + '/model.h5')
    
    test_data_dir = "Data/BRATS_2018/HGG_Testing"
    image = None
    seg_image = None
    for subdir in os.listdir(test_data_dir):
        for path in os.listdir(test_data_dir+ "/" + subdir):
            if mode in path:
                image = nib.load(test_data_dir + "/" + subdir + "/" + path).get_data()
                seg_image = nib.load(test_data_dir+ "/" + subdir + "/" + path.replace(mode, "seg")).get_data()
                break
    m = nmfComp.block_dim
    inds = [i for i in list(range(155)) if np.count_nonzero(seg_image[:,:,i]) > 0]

    for k in inds:
        seg_est = np.zeros(shape=(dataHandler.W, dataHandler.H))
        img = dataHandler.preprocess(image[:,:,k])
        _, H = nmfComp.run(img)
        H_cols = np.hsplit(H, H.shape[1])
        labels = [model.predict(x.T) for x in H_cols]
        #labels = model.predict(H.T)
        ind = 0
        for i in range(0, dataHandler.W, m):
            for j in range(0, dataHandler.H, m):
                seg_est[j:j+m, i:i+m] = np.full((m, m), np.argmax(labels[ind]))
                ind = ind+1
        
        fig = plt.figure()
        plt.gray();
        a=fig.add_subplot(1,3,1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Original')

        a=fig.add_subplot(1,3,2)
        plt.imshow(seg_image[:,:,k])
        plt.axis('off')
        plt.title('GT Segment')
        
        a=fig.add_subplot(1,3,3)
        plt.imshow(seg_est)
        plt.axis('off')
        plt.title('Estimate Segment')
        plt.show()
        
        
# evaluate the model

if __name__ == "__main__":
   main() 
