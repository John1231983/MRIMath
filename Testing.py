'''
Created on Aug 9, 2018

@author: daniel
'''
import sys
import os
from numpy import genfromtxt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.TimerModule import TimerModule
from Exploratory_Stuff.BlockDataHandler import BlockDataHandler
#from keras.callbacks import CSVLogger,ReduceLROnPlateau
from keras.layers import Dense, BatchNormalization, Conv1D
from keras.models import Sequential
from keras.callbacks import CSVLogger
#from keras.optimizers import SGD
#import os
from keras.models import load_model

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

from NMFComputer.SKNMFComputer import SKNMFComputer
import sys
import os
DATA_DIR = os.path.abspath("../")
sys.path.append(DATA_DIR)
def main():
    loss_data= genfromtxt("Models/2018-08-08_20_58_t1ce/model_loss_log.csv",delimiter=',')
    plt.figure()
    plt.plot(loss_data[:,1])
    #plt.plot(loss_data[:,1])
    plt.xlabel("Epochs") 
    plt.ylabel("Accuracy") 
    plt.title("Blocknet Loss Curve")
    plt.show()
    nmfComp = BasicNMFComputer(block_dim=8, num_components=100)
    dataHandler = BlockDataHandler("Data/BRATS_2018/HGG", nmfComp, num_patients = 25)
    mode = "t1ce"
    model = load_model("Models/2018-08-09_21_01_t1ce/model.h5")
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
        _, H = nmfComp.run(image[:,:,k])
        H_cols = np.hsplit(H, H.shape[1])
        labels = [model.predict(x.T) for x in H_cols]
        ind = 0
        for i in range(0, dataHandler.W, m):
            for j in range(0, dataHandler.H, m):
                label = 0
                if np.max(labels[ind]) > 0.95:
                    print(np.max(labels[ind]))
                    label = np.argmax(labels[ind])
                else:
                    label = 0
                seg_est[j:j+m, i:i+m] = np.full((m, m), label)
                ind = ind+1
        
        fig = plt.figure()
        plt.gray();
        a=fig.add_subplot(1,3,1)
        plt.imshow(image[:,:,k])
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
if __name__ == "__main__":
   main() 