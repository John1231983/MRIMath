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
from Exploratory_Stuff.SegNetDataHandler import SegNetDataHandler
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


def dice_and_iou(y_true, y_pred):
    alpha = 0.9
    beta = 1 - alpha
    dice = dice_coef_loss(y_true, y_pred)
    iou = iou_loss(y_true, y_pred)
    return alpha*dice + beta*iou

def computeDice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dice = 2. * intersection.sum() / (im1.sum() + im2.sum())
    if math.isnan(dice):
        return 0
    return dice

def computeDice2(im1, im2):


    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = im1 * im2
    dice = 2. * intersection.sum() / (im1.sum() + im2.sum())
    if math.isinf(dice):
        return 0
    return dice
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return 1. * intersection / union

def iou_loss(y_true, y_pred):
    return 1 - iou(y_true, y_pred)

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
    dataHandler = SegNetDataHandler("Data/BRATS_2018/HGG", num_patients = num_training_patients, modes = ["flair"])
    dataHandler.loadData()
    dataHandler.preprocessForNetwork()
    x_test = dataHandler.X
    x_seg_test = dataHandler.labels
    dataHandler.clear()
    

    input_shape = (dataHandler.W,dataHandler.H, len(modes))
    

    segnet = load_model("Models/segnet_2018-10-03-20:51/model.h5", custom_objects={'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D, 'MaxUnpooling2D':MaxUnpooling2D, 'dice_coef_loss': dice_coef_loss, 'dice_coef':dice_coef, 'dice_and_iou': dice_and_iou, 'iou_loss': iou_loss, 'iou':iou})
    
    decoded_imgs = segnet.predict(x_test)
      
    avg_dice = 0
    N = len(decoded_imgs)
    for i in range(len(decoded_imgs)):
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
        plt.imshow(decoded_imgs[i].reshape(dataHandler.W, dataHandler.W))
        plt.axis('off')
        plt.title('Predicted Segment')

        plt.show()
    


if __name__ == "__main__":
   main() 
