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
import matplotlib.pyplot as plt

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
import SimpleITK as sitk

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
    
    def preprocess(self, image):
        sitk_image = sitk.GetImageFromArray(image)
        sitk_image = sitk.IntensityWindowing(sitk_image, np.percentile(image, 1), np.percentile(image, 99))
        sitk_image = sitk.Cast( sitk_image, sitk.sitkFloat64 )

        corrected_image = sitk.N4BiasFieldCorrection(sitk_image, sitk_image > 0);
        corrected_image = sitk.GetArrayFromImage(corrected_image)
        # corrected_image = exposure.equalize_hist(corrected_image)
        return corrected_image
        
        
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
                    seg_image = nib.load(self.dataDirectory + "/" + subdir + "/" + path.replace(mode, "seg")).get_data()
                    with Pool(processes=8) as pool:
                        Y = pool.map(partial(self.performNMFOnSlice, image, seg_image), list(range(155)))
                    self.X.extend(Y)
                """
                elif "seg" in path:
                    seg_image = nib.load(self.dataDirectory + "/" + subdir + "/" + path).get_data()
                    for i in range(155):
                        self.labels.extend(seg_image[:,:,i])
                """
            J = J + 1
        self.preprocessForNetwork()


    def performNMFOnSlice(self, image, seg_image, i):
        W, H = self.nmfComp.run(image[:,:,i])
        self.processData(image[:,:,i], W,H, seg_image[:,:,i])
        return W, H
    
    
    def processData(self, image, W, H, seg_image):
        
        indices = np.argmax(W, axis=0)
        #H = H[indices > 0]
        regions = np.argmax(H, axis=0)
        n = 6
        region_5 = regions.copy()
        region_5_and_6 = regions.copy()
        region_5_and_6_and_7 = regions.copy()
        
        region_5[regions < 3] = 0
        region_5[regions > 3] = 0
        region_5 = region_5.astype(bool)
        
        region_5_and_6[regions < 3] = 0
        region_5_and_6[regions > 4] = 0
        region_5_and_6 = region_5_and_6.astype(bool)

        
        #region_5_and_6_and_7[regions < ] = 0
        region_5_and_6_and_7[regions < 6] = 1
        region_5_and_6_and_7[regions > 6] = 0

        #region_5_and_6_and_7[regions > 6] = 1

        region_5_and_6_and_7 = region_5_and_6_and_7.astype(bool)
        
        
        reg_5_image = image.copy()
        reg_5_and_6_image = image.copy()
        reg_5_and_6_and_7_image = image.copy()
        #reg_4_and_5_and_6_and_7_image = image.copy()

        regions = regions.astype(bool)
        m = self.nmfComp.block_dim
        ind = 0
        for i in range(0, image.shape[0], m):
            for j in range(0, image.shape[1], m):
                reg_5_image[i:i+m, j:j+m] *= region_5[ind]
                reg_5_and_6_image[i:i+m, j:j+m] *= region_5_and_6[ind] 
                reg_5_and_6_and_7_image[i:i+m, j:j+m] *= region_5_and_6_and_7[ind]
                ind = ind + 1
                
        fig = plt.figure()
        plt.gray();   
        a=fig.add_subplot(1,5,1)
        plt.imshow(image)
        plt.axis('off')
        a=fig.add_subplot(1,5,2)
        plt.imshow(reg_5_image)
        plt.axis('off')
        a=fig.add_subplot(1,5,3)
        plt.imshow(reg_5_and_6_image)
        plt.axis('off')
        a=fig.add_subplot(1,5,4)
        plt.imshow(reg_5_and_6_and_7_image)
        plt.axis('off')
        a=fig.add_subplot(1,5,5)
        plt.imshow(seg_image)
        plt.axis('off')

        plt.show()
                
                    
        

    def preprocessForNetwork(self):

        """
        plt.bar(np.arange(H.shape[0]), H[:,10])
            plt.show()
         """
            
            
           

