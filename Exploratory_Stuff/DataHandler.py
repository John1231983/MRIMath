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
from keras.utils import to_categorical

from scipy import stats

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
    
    def __init__(self, dataDirectory, nmfComp, W = 240, H = 240):
        self.dataDirectory = dataDirectory
        self.X = []
        self.labels = []
        self.W = W
        self.H = H
        self.nmfComp = nmfComp
    
    def preprocess(self, image):
        sitk_image = sitk.GetImageFromArray(image)
        sitk_image = sitk.IntensityWindowing(sitk_image, np.percentile(image, 1), np.percentile(image, 99))
        sitk_image = sitk.Cast( sitk_image, sitk.sitkFloat64 )

        corrected_image = sitk.N4BiasFieldCorrection(sitk_image, sitk_image > 0);
        corrected_image = sitk.GetArrayFromImage(corrected_image)
        # corrected_image = exposure.equalize_hist(corrected_image)
        return corrected_image
        
    def clear(self):
        self.X = []
        self.labels = []
        
    def loadData(self, mode):
        timer = TimerModule()
        timer.startTimer()
        J = 0
        for subdir in os.listdir(self.dataDirectory):
            if J > 1:
                timer.stopTimer()
                print(timer.getElapsedTime())
                break
            #Y = []
            for path in os.listdir(self.dataDirectory + "/" + subdir):
                if mode in path:
                    image = nib.load(self.dataDirectory + "/" + subdir + "/" + path).get_data()
                    seg_image = nib.load(self.dataDirectory + "/" + subdir + "/" + path.replace(mode, "seg")).get_data()
                    inds = [i for i in list(range(155)) if np.count_nonzero(seg_image[:,:,i]) > 0]
                    with Pool(processes=8) as pool:
                        temp = pool.map(partial(self.performNMFOnSlice, image, seg_image), inds)
                    foo = [i[0] for i in temp]
                    bar = [i[1] for i in temp]

                    self.X.extend([item for sublist in foo for item in sublist])
                    self.labels.extend([item for sublist in bar for item in sublist])
                    J = J + 1
                    break

    
    def setDataDirectory(self, dataDirectory):
        self.dataDirectory = dataDirectory
        
    def getImagesWithSegment(self, seg_image, i):
        if np.count_nonzero(seg_image[:,:,i]) > 0:
            self.labels.append(seg_image[:,:,i])
            return i
        return -1

    def performNMFOnSlice(self, image, seg_image, i):
        W, H = self.nmfComp.run(image[:,:,i])
        return self.processData2(W,H, seg_image[:,:,i])
    
    def processData2(self, W, H, seg_image):
        X = []
        y = []
        m = self.nmfComp.block_dim
        max_background_blocks = 0.05*H.shape[1]
        num_background_blocks = 0
        ind = 0
        for i in range(0, seg_image.shape[0], m):
            for j in range(0, seg_image.shape[1], m):
                counts = np.bincount(seg_image[i:i+m, j:j+m].flatten())
                mode = int(np.argmax(counts))
                if mode == 0:
                    if num_background_blocks < max_background_blocks:
                        y.append(mode)
                        X.append(H[:, ind:ind+1])
                    num_background_blocks = num_background_blocks + 1
                else:
                    y.append(mode)
                    X.append(H[:, ind:ind+1])
                ind = ind + 1
                        

        return X, y

    
    
    def processData(self, image, W, H, seg_image):
        X = []
        y = []
        indices = np.argmax(W, axis=0)
        #H = H[indices > 0]
        regions = np.argmax(H, axis=0)
        
        
        
        region_1 = regions.copy()
        region_1_and_2 = regions.copy()
        region_1_and_2_and_3 = regions.copy()
        region_1_and_2_and_3_and_4 = regions.copy()
        region_1_and_2_and_3_and_4_and_5 = regions.copy()
        region_1_and_2_and_3_and_4_and_5_and_6 = regions.copy() 
        
        region_1[regions < 0] = 0
        region_1[regions > 1] = 0
        region_1 = region_1.astype(bool)
        
        region_1_and_2[regions < 1] = 0
        region_1_and_2[regions > 2] = 0
        region_1_and_2 = region_1_and_2.astype(bool)
                
        region_1_and_2_and_3[regions < 1] = 0
        region_1_and_2_and_3[regions > 3] = 0
        region_1_and_2_and_3 = region_1_and_2_and_3.astype(bool)
        
        region_1_and_2_and_3_and_4[regions < 1] = 0
        region_1_and_2_and_3_and_4[regions > 4] = 0
        region_1_and_2_and_3_and_4 = region_1_and_2_and_3_and_4.astype(bool)
        
        region_1_and_2_and_3_and_4_and_5[regions < 1] = 0
        region_1_and_2_and_3_and_4_and_5[regions > 5] = 0
        region_1_and_2_and_3_and_4_and_5 = region_1_and_2_and_3_and_4_and_5.astype(bool)

        
        region_1_and_2_and_3_and_4_and_5_and_6[regions < 1] = 0
        region_1_and_2_and_3_and_4_and_5_and_6[regions > 6] = 0
        region_1_and_2_and_3_and_4_and_5_and_6 = region_1_and_2_and_3_and_4_and_5_and_6.astype(bool)
        
        reg_1_image = image.copy()
        reg_1_and_2_image = image.copy()
        reg_1_and_2_and_3_image = image.copy()
        reg_1_and_2_and_3_and_4_image = image.copy()
        reg_1_and_2_and_3_and_4_and_5_image = image.copy()
        reg_1_and_2_and_3_and_4_and_5_and_6_image = image.copy()
        
        #regions = regions.astype(bool)
        m = self.nmfComp.block_dim
        ind = 0
        for i in range(0, seg_image.shape[0], m):
            for j in range(0, seg_image.shape[1], m):
                reg_1_image[i:i+m, j:j+m] *= region_1[ind]
                reg_1_and_2_image[i:i+m, j:j+m] *= region_1_and_2[ind]
                reg_1_and_2_and_3_image[i:i+m, j:j+m] *= region_1_and_2_and_3[ind]
                reg_1_and_2_and_3_and_4_image[i:i+m, j:j+m] *= region_1_and_2_and_3_and_4[ind]
                reg_1_and_2_and_3_and_4_and_5_image[i:i+m, j:j+m] *= region_1_and_2_and_3_and_4_and_5[ind] 
                reg_1_and_2_and_3_and_4_and_5_and_6_image[i:i+m, j:j+m] *= region_1_and_2_and_3_and_4_and_5_and_6[ind]
                ind = ind + 1
                """
        #seg_image[seg_image > 0] = 1
        """
        X.append(reg_1_and_2_and_3_image)
        y.append(seg_image)
        
        X.append(reg_1_and_2_and_3_and_4_image)
        y.append(seg_image)
        
        X.append(reg_1_and_2_and_3_and_4_and_5_image)
        y.append(seg_image)

        X.append(reg_1_and_2_and_3_and_4_and_5_and_6_image)
        y.append(seg_image)
        
        
        
        """
        fig = plt.figure()
        plt.gray();   
        
        a=fig.add_subplot(1,8,1)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Original')
        
        a=fig.add_subplot(1,8,2)
        plt.imshow(reg_1_image)
        plt.axis('off')
        plt.title('1')
        
        
        a=fig.add_subplot(1,8,3)
        plt.imshow(reg_1_and_2_image)
        plt.axis('off')
        plt.title(r'1 $\cup$ 2')
        
        a=fig.add_subplot(1,8,4)
        plt.imshow(reg_1_and_2_and_3_image)
        plt.axis('off')
        plt.title(r'1 $\cup$ 2 $\cup$ 3')
        self.X.append(reg_1_and_2_and_3_image)
        
        
        a=fig.add_subplot(1,8,5)
        plt.imshow(reg_1_and_2_and_3_and_4_image)
        plt.axis('off')
        plt.title(r'1 $\cup$ 2 $\cup$ 3 $\cup$ 4')
        
        a=fig.add_subplot(1,8,6)
        plt.imshow(reg_1_and_2_and_3_and_4_and_5_image)
        plt.axis('off')
        plt.title(r'1 $\cup$ 2 $\cup$ 3 $\cup$ 4 $\cup$ 5')


        a=fig.add_subplot(1,8,7)
        plt.imshow(reg_1_and_2_and_3_and_4_and_5_and_6_image)
        plt.axis('off') 
        plt.title(r'1 $\cup$ 2 $\cup$ 3 $\cup$ 4 $\cup$ 5 $\cup$ 6')

        a=fig.add_subplot(1,8,8)
        plt.imshow(seg_image)
        plt.axis('off')
        plt.title('Segment')

        plt.show()
        """
        return X, y

    
    
                
                    
    

    def preprocessForNetwork(self):
        n_imgs = len(self.X)
        
        self.X = np.array( self.X )
        self.X = self.X.reshape(n_imgs,self.nmfComp.block_dim)
        #self.labels = np.array( self.labels )
        self.labels = np_utils.to_categorical(self.labels)
        #self.labels = self.labels.reshape(n_imgs,self.W,self.H,1)
        # self.labels = self.labels.reshape(n_imgs, self.W*self.H,2)

        
        
            
            
           

