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
from sklearn.utils.extmath import cartesian
import math


sess = tf.Session()
K.set_session(sess)
g = K.get_session().graph
W = 128
H = 128
all_img_locations = tf.convert_to_tensor(cartesian([np.arange(W), np.arange(H)]), dtype=tf.float32)
n_pixels = W *H
eps = 1e-6
alpha = 2
max_dist = math.sqrt(W**2 + H**2)

def dice_coef(y_true, y_pred, smooth=1e-3):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice
    

def dice_coef_loss(y_true, y_pred):
    return -tf.log(dice_coef(y_true, y_pred))

def dice_coef_multilabel(y_true, y_pred, numLabels=3):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,index], y_pred[:,:,index])
    return dice/numLabels




def dice_coef_reg_1(y_true, y_pred):
    dice = dice_coef(y_true[:,:,0], y_pred[:,:,0])
    return dice


def dice_coef_reg_2(y_true, y_pred):
    dice = dice_coef(y_true[:,:,1], y_pred[:,:,1])
    return dice

def dice_coef_reg_3(y_true, y_pred):
    dice = dice_coef(y_true[:,:,2], y_pred[:,:,2])
    return dice


def dice_coef_multilabel_loss(y_true, y_pred):
    return -tf.log(dice_coef_multilabel(y_true, y_pred))

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
        #y[i,:,:] = cv2.Canny(np.uint8(y[i,:,:]),0,1)
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

def cdist (A, B):  

    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)
    
    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])
    
    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))
    return D
      


def hausdorff_dist_loss(y_true, y_pred):
    batched_losses = tf.map_fn(lambda x: 
                hausdorff_dist(x[0], x[1]), 
                (y_true, y_pred), 
                dtype=tf.float32)
    return K.mean(tf.log(tf.stack(batched_losses)))
    
def hausdorff_dist(y_true, y_pred):
        
        y_true = K.reshape(y_true, [W,H])
        gt_points = K.cast(tf.where(y_true > 0.5), dtype = tf.float32)
        num_gt_points = tf.shape(gt_points)[0]
        
        y_pred = K.flatten(y_pred)
        p = y_pred
        p_replicated = tf.squeeze(K.repeat(tf.expand_dims(p,axis=-1), num_gt_points))
        
        d_matrix = cdist(all_img_locations, gt_points)
        num_est_pts = tf.reduce_sum(p)
        term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))
        
        
        d_div_p = K.min((d_matrix + eps) / (p_replicated**alpha + (eps / max_dist)), 0)
        d_div_p = K.clip(d_div_p, 0, max_dist)
        term_2 = K.mean(d_div_p, axis=0) 
        
        return term_1 + term_2

def avg_hausdorff_distance(y_true, y_pred):
    batched_losses = tf.map_fn(lambda x: 
            hausdorff_dist(x[0], x[1]), 
            (y_true, y_pred), 
            dtype=tf.float32)
    return K.mean(tf.stack(batched_losses))
    
def combinedHausdorffAndDice(y_true,y_pred):
    alpha = 0.5
    beta = 1 - alpha
    dice = dice_coef_loss(y_true, y_pred)
    hd = hausdorff_dist_loss(y_true, y_pred)
    return alpha*dice + beta*hd

def custom_sigmoid(t, n_labels = 3):
    foo = tf.unstack(t, axis=3)
    result = []
    for i in range(n_labels):
        result.append(K.sigmoid(foo[i]))
    return K.concatenate(result, axis=0)
