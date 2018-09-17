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
from skimage import exposure

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
    num_patients = 0
    load_mode = None
    def __init__(self, dataDirectory, nmfComp, W = 100, H = 100, num_patients = 3, load_mode = "training"):
        self.dataDirectory = dataDirectory
        self.X = []
        self.labels = []
        self.W = W
        self.H = H
        self.nmfComp = nmfComp
        self.setNumPatients(num_patients)
        self.setLoadingMode(load_mode)
    
    def setNumPatients(self, num_patients):
        if num_patients > 0:
            self.num_patients = num_patients
            
    def preprocess(self, image):
        
        sitk_image = sitk.GetImageFromArray(image)
        #sitk_image = sitk.IntensityWindowing(sitk_image, np.percentile(image, 1), np.percentile(image, 99))
        sitk_image = sitk.Cast( sitk_image, sitk.sitkFloat64 )

        corrected_image = sitk.N4BiasFieldCorrection(sitk_image, sitk_image > 0);
        corrected_image = sitk.GetArrayFromImage(corrected_image)
        #corrected_image = exposure.equalize_hist(corrected_image)
        return corrected_image
        
    def clear(self):
        self.X = []
        self.labels = []
        
    def loadData(self, mode):
        pass
    
    def setLoadingMode(self, load_mode):
        if load_mode is "training" or load_mode is "validation" or load_mode is "testing":
            self.load_mode = load_mode
        else:
            self.load_mode = "training"
    
    def setDataDirectory(self, dataDirectory):
        self.dataDirectory = dataDirectory
        
    def getImagesWithSegment(self, seg_image, i):
        if np.count_nonzero(seg_image[:,:,i]) > 0:
            self.labels.append(seg_image[:,:,i])
            return i
        return -1

    def performNMFOnSlice(self, image, seg_image, i):
        pass
    
    def processData(self, W, H, seg_image):
        pass
    
    def preprocessForNetwork(self):
        pass
    
    def bbox(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin,rmax, cmin,cmax


    
    

        
        
            
            
           

