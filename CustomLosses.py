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


sess = tf.Session()
K.set_session(sess)
g = K.get_session().graph

def dice_coef(y_true, y_pred, smooth=1):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    #dice = K.print_tensor(dice, "Dice is: ")
    return dice
    


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return 1. * intersection / union

def iou_loss(y_true, y_pred):
    return 1 - iou(y_true, y_pred)


def dice_and_iou(y_true, y_pred):
    alpha = 0.9
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
    rnd_name = name + "_" + 'PyFuncGrad' + 'ABC@a1b2c3'

    # Register Tensorflow Gradient
    tf.RegisterGradient(rnd_name)(grad)

    # Get current graph
    #g = tf.get_default_graph()

    # Add gradient override map
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
    
def showContours(a):
    fig, ax = plt.subplots()
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
    
    #y_true.set_shape(y_true_shape)
    y_true.set_shape((None, 128, 128))
    
    y_pred = py_func(computeContour, 
            [y_pred], 
            [tf.float32], 
            name = "contour", 
            grad=_ContourGrad)[0]
    #y_pred.set_shape(y_pred_shape)
    y_pred.set_shape((None, 128, 128))

    hausdorff = y_pred * y_true
    print(hausdorff.get_shape())
    hausdorff = tf.Print(hausdorff, [hausdorff], summarize=10)

    hausdorff = tf.map_fn(lambda x: K.min(x), hausdorff, dtype=tf.float32)
    print(hausdorff.get_shape())
    hausdorff = tf.Print(hausdorff, [hausdorff], summarize=10)

    hausdorff = K.max(hausdorff)
    print(hausdorff.get_shape())
    hausdorff = tf.Print(hausdorff, [hausdorff], summarize=10)


    return hausdorff


        
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
    
    y_true.set_shape(y_true_shape)
    #y_true.set_shape((None, 128, 128))

    y_pred = py_func(computeContour, 
            [y_pred], 
            [tf.float32], 
            name = "contour", 
            grad=_ContourGrad)[0]
    y_pred.set_shape(y_pred_shape)
    #y_pred.set_shape((None, 128, 128))
    
    finalChamferDistanceSum = y_pred * y_true
    finalChamferDistanceSum = tf.map_fn(lambda x: 
                                        K.sum(x), 
                                        finalChamferDistanceSum, 
                                        dtype=tf.float32)
    

    #finalChamferDistanceSum = tf.nn.l2_normalize(finalChamferDistanceSum)

    finalChamferDistanceSum = K.mean(finalChamferDistanceSum)

    #finalChamferDistanceSum = tf.Print(finalChamferDistanceSum, [finalChamferDistanceSum], summarize=10)
    return finalChamferDistanceSum



def chamfer_loss(y_true, y_pred):   
    return chamfer_dist(y_true, y_pred)

def combinedHausdorffAndDice(y_pred, y_true):
    alpha = 0.5
    beta = 1 - alpha
    dice = dice_coef_loss(y_true, y_pred)
    hd = hausdorff_dist(y_true, y_pred)
    return alpha*dice + beta*hd

def combinedDiceAndChamfer(y_pred, y_true):
    alpha = 0.5
    beta = 1 - alpha
    dice = dice_coef_loss(y_true, y_pred)
    cd = chamfer_dist(y_true, y_pred)
    return alpha*dice + beta*cd