'''
Created on Oct 12, 2018

@author: daniel
'''

import cv2
from skimage import measure
from scipy.ndimage.morphology import distance_transform_edt
import tensorflow as tf
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
import random


sess = tf.Session()
K.set_session(sess)
g = K.get_session().graph

def dice_coef(y_true, y_pred, smooth=1e-3):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice
    


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def dice_coef_multilabel(y_true, y_pred, numLabels=4):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,index], y_pred[:,:,index])
        
    return dice/numLabels

def dice_coef_multilabel_loss(y_true, y_pred):
    return 1 - dice_coef_multilabel(y_true, y_pred)

def iou(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return 1. * intersection / union

def iou_loss(y_true, y_pred):
    return 1 - iou(y_true, y_pred)


def dice_and_iou(y_true, y_pred):
    alpha = 0.5
    beta = 1 - alpha
    dice = dice_coef_loss(y_true, y_pred)
    iou = iou_loss(y_true, y_pred)
    return alpha*dice + beta*iou

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    """
    PyFunc defined as given by Tensorflow
    :param func: Custom Function
    :param inp: Function Inputs
    :param Tout: Ouput Type of out Custom Function
    :param stateful: Calculate Gradients when stateful is True
    :param name: Name of the PyFunction
    :param grad: Custom Gradient Function
    :return:
    """
    # Generate Random Gradient name in order to avoid conflicts with inbuilt names
    rnd_name = name + "_" + 'PyFuncGrad' + str(random.randint(1,100))

    # Register Tensorflow Gradient
    tf.RegisterGradient(rnd_name)(grad)

    # Get current graph
    #g = tf.get_default_graph()

    # Add gradient override map
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
    
def showContours(a):
    _, ax = plt.subplots()
    ax.imshow(a, interpolation='nearest', cmap=plt.cm.gray)
    contours = measure.find_contours(a, 0.5)
    print(contours)
    for _, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    


def computeEDT(y):
    for i in range(y.shape[0]):
        #print(y.shape)
        y[i,:,:] = cv2.Canny(np.uint8(y[i,:,:]),0,1)
        y[i,:,:] = distance_transform_edt(np.logical_not(y[i,:,:]))
    return y

def computeContour(y):
    for i in range(y.shape[0]):
        y[i,:,:][y[i,:,:] < 0.5] = 0
        y[i,:,:][y[i,:,:] > 0.5] = 1
        y[i,:,:] = cv2.Canny(np.uint8(y[i,:,:]),0,1)
        #cv2.imshow("",y[i,:,:])
        #scv2.waitKey(0)
    return y

def _EDTGrad(op, grad):
    return 0*op.inputs[0]

def _ContourGrad(op, grad):
    return 0*op.inputs[0]   

def hausdorff_dist(y_true, y_pred):
    
    y_true = K.reshape(y_true, [K.tf.shape(y_true)[0],128,128])
    y_pred = K.reshape(y_pred, [K.tf.shape(y_pred)[0],128,128])
    y_true_shape = y_true.get_shape()
    y_pred_shape = y_pred.get_shape()
    
    y_true = py_func(computeEDT, 
                [y_true], 
                [tf.float32], 
                name = "edt", 
                grad=_EDTGrad)[0]
    
    y_pred = py_func(computeContour, 
            [y_pred], 
            [tf.float32], 
            name = "contour", 
            grad=_ContourGrad)[0]
    
    y_true.set_shape(y_true_shape)
    y_pred.set_shape(y_pred_shape)

    hausdorffDistance = y_pred * y_true
    hausdorffDistance = tf.map_fn(lambda x: 
                                        K.max(x), 
                                        hausdorffDistance, 
                                        dtype=tf.float32)

    #hausdorffDistance = tf.nn.l2_normalize(hausdorffDistance)
    hausdorffDistance = K.mean(hausdorffDistance)
    
    return hausdorffDistance

def hausdorff_dist_multilabel(y_true, y_pred, numLabels=4):
    d_h=0
    for index in range(numLabels):
        d_h += hausdorff_dist(y_true[:,:,index], y_pred[:,:,index])
    return d_h

        
def chamfer_dist(y_true, y_pred):
    
    y_true = K.reshape(y_true, [K.tf.shape(y_true)[0],128,128])
    y_pred = K.reshape(y_pred, [K.tf.shape(y_pred)[0],128,128])
    y_true_shape = y_true.get_shape()
    y_pred_shape = y_pred.get_shape()
    
    y_true = py_func(computeEDT, 
                [y_true], 
                [tf.float32], 
                name = "edt", 
                grad=_EDTGrad)[0]

    y_pred = py_func(computeContour, 
            [y_pred], 
            [tf.float32], 
            name = "contour", 
            grad=_ContourGrad)[0]
            
    y_true.set_shape(y_true_shape)
    y_pred.set_shape(y_pred_shape)
    
    finalChamferDistanceSum = y_pred * y_true
    finalChamferDistanceSum = tf.map_fn(lambda x: 
                                        K.sum(x), 
                                        finalChamferDistanceSum, 
                                        dtype=tf.float32)
    

    finalChamferDistanceSum = tf.nn.l2_normalize(finalChamferDistanceSum)
    finalChamferDistanceSum = K.mean(finalChamferDistanceSum)

    return finalChamferDistanceSum



def chamfer_dist_multilabel(y_true, y_pred, numLabels=4):
    d_cd=0
    for index in range(numLabels):
        d_cd += chamfer_dist(y_true[:,:,index], y_pred[:,:,index])
    return d_cd


def chamfer_loss(y_true, y_pred):   
    return chamfer_dist(y_true, y_pred)

def combinedHausdorffAndDice(y_pred, y_true):
    alpha = 0.5
    beta = 1 - alpha
    dice = dice_coef_loss(y_true, y_pred)
    hd = hausdorff_dist(y_true, y_pred)
    return alpha*dice + beta*hd

def combinedHausdorffAndDiceMultilabel(y_pred, y_true):
    alpha = 0.5
    beta = 1 - alpha
    dice = dice_coef_multilabel_loss(y_true, y_pred)
    hd = hausdorff_dist_multilabel(y_true, y_pred)
    return alpha*dice + beta*hd

def combinedDiceAndChamfer(y_pred, y_true):
    alpha = 0.5
    beta = 1 - alpha
    dice = dice_coef_loss(y_true, y_pred)
    cd = chamfer_dist(y_true, y_pred)
    return alpha*dice + beta*cd
    
    
def combinedDiceAndChamferMultilabel(y_pred, y_true):
    alpha = 0.5
    beta = 1 - alpha
    dice = dice_coef_multilabel_loss(y_true, y_pred)
    cd = chamfer_dist_multilabel(y_true, y_pred)
    return alpha*dice + beta*cd


