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
from DataHandlers.SegNetDataHandler import SegNetDataHandler
#from keras.callbacks import CSVLogger,ReduceLROnPlateau
from keras.layers import Conv2D, Activation, MaxPooling2D, Reshape, Dense, Flatten, BatchNormalization, Dropout, LeakyReLU, PReLU,concatenate
from keras.models import Sequential, Model, Input
#from keras.optimizers import SGD
#import os
from Utils.EmailHandler import EmailHandler
from Utils.HardwareHandler import HardwareHandler
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from createSegNet import createSegNet
from createSegnetWithIndexPooling import createSegNetWithIndexPooling
from keras.models import load_model
from Mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D
import math
from CustomLosses import combinedDiceAndChamfer
from CustomLosses import dice_coef


#from keras.utils.training_utils import multi_gpu_model
#import keras.backend as K
#import tensorflow as tf
from keras.optimizers import SGD
from keras.callbacks import CSVLogger

from NMFComputer.BasicNMFComputer import BasicNMFComputer
import cv2
import sys
import os
DATA_DIR = os.path.abspath("../")
sys.path.append(DATA_DIR)

def computeDice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
 
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)
    dice = 2. * intersection.sum() / (im1.sum() + im2.sum())
    if math.isnan(dice):
        return 0
    return dice
def main():

    hardwareHandler = HardwareHandler()
    emailHandler = EmailHandler()
    timer = TimerModule()
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d-%H:%M')
    print(date_string)
    
    num_training_patients = 30
    num_validation_patients = 3
    
    modes = ["flair"]
    dataHandler = SegNetDataHandler("Data/BRATS_2018/HGG_Testing", num_patients = num_training_patients, modes = ["flair"])
    dataHandler.loadData()
    dataHandler.preprocessForNetwork()
    x_test = dataHandler.X
    x_seg_test = dataHandler.labels
    dataHandler.clear()
    

    input_shape = (dataHandler.W,dataHandler.H, len(modes))
    

    segnet = load_model("Models/segnet_2018-10-16-10:42/model.h5", custom_objects={'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D, 'MaxUnpooling2D':MaxUnpooling2D, 'combinedDiceAndChamfer':combinedDiceAndChamfer, 'dice_coef':dice_coef})
    
    decoded_imgs = segnet.predict(x_test)
    avg_dice = 0
    #N = len(decoded_imgs)
    N = 100
    for i in range(N):
            decoded_imgs[i][decoded_imgs[i] < 0.5] = 0
            kernel = np.ones((3,3),np.uint8)
            decoded_imgs[i] = cv2.morphologyEx(decoded_imgs[i],cv2.MORPH_OPEN,kernel)
            dice = computeDice(x_seg_test[i], np.squeeze(decoded_imgs[i]))
            avg_dice = avg_dice + dice
    print(str(avg_dice/N))
    n = 100
    for i in range(n):
        fig = plt.figure()
        plt.gray();   
        a=fig.add_subplot(1,3,1)
        plt.imshow(x_test[i,:,:,0])
        plt.axis('off')
        plt.title('Original')
        
        a=fig.add_subplot(1,3,2)
        plt.imshow(x_seg_test[i].reshape(dataHandler.W, dataHandler.W))
        plt.axis('off')
        plt.title('GT Segment')
        
        a=fig.add_subplot(1,3,3)
        decoded_imgs[i][decoded_imgs[i] > 0.5] = 1
        decoded_imgs[i][decoded_imgs[i] < 0.5] = 0

        plt.imshow(decoded_imgs[i].reshape(dataHandler.W, dataHandler.W))
        plt.axis('off')
        plt.title('Predicted Segment')

        plt.show()
    


if __name__ == "__main__":
   main() 
