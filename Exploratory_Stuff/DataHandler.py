'''

Class designed to do all data handling and manipulation, ranging from dataloading to network preprocessing.
As time goes on, some of this may be refactored, and some of this functionality is contingent on the data being stored 
in a certain structure. 

@author Daniel Enrico Cahall

'''

"""
import os
import cv2
from functools import partial
import threading
import random
import numpy as np
"""
from math import floor
from multiprocessing import Pool
from keras.utils import np_utils
from Utils.TimerModule import TimerModule
#from keras.callbacks import CSVLogger,ReduceLROnPlateau
#from keras.layers import Conv2D, Activation, MaxPooling2D, Reshape, Dense, Flatten, BatchNormalization, Dropout, LeakyReLU, PReLU,concatenate
#from keras.models import Sequential, Model, Input
#from keras.optimizers import SGD
import os
from datetime import datetime
#from keras.utils.training_utils import multi_gpu_model
#import keras.backend as K
#import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import nibabel as nib
import sys
from functools import partial
from NMFComputer.NMFComputer import NMFComputer

timer = TimerModule()
now = datetime.now()
date_string = now.strftime('%Y-%m-%d_%H_%M')
DATA_DIR = os.path.abspath("../")
sys.path.append(DATA_DIR)
class DataHandler:
    
    W = None
    H = None
    dataDirectory = None
    X = []
    labels = []
    nmfComp = None
    
    def __init__(self, dataDirectory, nmfComp):
        self.dataDirectory = dataDirectory
        self.X = []
        self.labels = []
        self.nmfComp = nmfComp
        
    def loadData(self, mode):
        timer = TimerModule()
        timer.startTimer()
        J = 0
        for subdir in os.listdir(self.dataDirectory):
            if J > 1:
                timer.stopTimer()
                print(timer.getElapsedTime())
                break
            Y = []
            for path in os.listdir(self.dataDirectory + "/" + subdir):
                if mode in path:
                    image = nib.load(self.dataDirectory + "/" + subdir + "/" + path).get_data()
                    with Pool(processes=8) as pool:
                        Y = pool.map(partial(self.performNMFOnSlice, image), list(range(155)))
                    self.X.extend(Y)
            J = J + 1
            """

            J = J + 1
        self.preprocessForNetwork()
"""
    def performNMFOnSlice(self, image, i):
        W, H = self.nmfComp.run(image[:,:,i])
        return W, H
    
        
    def getLabel(self, mat, i):
        image = mat[:,:,i]
        labels = []
        for r in range(0,image.shape[0], self.nmfComp.row_window_size):
            for c in range(0,image.shape[1], self.nmfComp.col_window_size):
                window = image[r:r+self.nmfComp.row_window_size,c:c+self.nmfComp.col_window_size]
                labels.append(window[floor(self.nmfComp.row_window_size/2), floor(self.nmfComp.col_window_size/2)])
        return labels
                    

    def preprocessForNetwork(self):
        print(len(self.labels))
        print(len(self.X))
        self.X = np.array(self.X)
        self.X = self.X.transpose()
        self.labels = np.array(self.labels)
        self.labels = np_utils.to_categorical(self.labels)
           

