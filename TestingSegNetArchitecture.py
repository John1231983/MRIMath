'''
Created on Jul 10, 2018

@author: daniel
'''

#from multiprocessing import Process, Manager
#from keras.utils import np_utils
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.spatial import distance
#tf.enable_eager_execution()
from scipy.spatial.distance import directed_hausdorff
import sys
import os
from dask.array.tests.test_numpy_compat import dtype
from dask.array.routines import nonzero
from TestingUNetArchitecture import combinedHausdorffAndDice
from tensorflow.python.framework import ops

from Utils.TimerModule import TimerModule
from Exploratory_Stuff.SegNetDataHandler import SegNetDataHandler
#from keras.callbacks import CSVLogger,ReduceLROnPlateau
#from keras.optimizers import SGD
#import os
from Utils.HardwareHandler import HardwareHandler
from keras.models import Model
from keras import backend as K
import numpy as np
from datetime import datetime
from createSegNet import createSegNet
from createSegnetWithIndexPooling import createSegNetWithIndexPooling
#from keras.utils.training_utils import multi_gpu_model
#import keras.backend as K
#import tensorflow as tf
from keras.optimizers import SGD
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator

from NMFComputer.BasicNMFComputer import BasicNMFComputer
from Canny_Tensorflow import TF_Canny
import sys
import os
import cv2
from skimage import measure
from scipy.ndimage.morphology import distance_transform_edt
np.set_printoptions(threshold=np.inf)

DATA_DIR = os.path.abspath("../")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(DATA_DIR)


sess = tf.Session()
K.set_session(sess)
g = K.get_session().graph


def dice_coef(y_true, y_pred, smooth=1e-3):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


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
def computeHausdorff(y_pred, y_true):
    hausdorff_distances = []
    hausdorff_distances_prime = []
    for i in range(y_true.shape[0]):
        a = np.squeeze(y_true[i,:,:]);
        b = np.squeeze(y_pred[i,:,:]);
        #showContours(b)
        #true_contours = measure.find_contours(a, 0.5)
        #pred_contours = measure.find_contours(b, 0.5)
        a = np.argwhere(a > 0.5)
        b = np.argwhere(b > 0.5)
        hausdorff_data_ab = directed_hausdorff(a, b)
        hausdorff_data_ba = directed_hausdorff(b, a)
        h_ab = hausdorff_data_ab[0]
        h_ba = hausdorff_data_ba[0]
        hausdorff_distances.append(np.float32(max(h_ab, h_ba)))
        hausdorff_distances_prime.append(np.float32(-1))
    hausdorff_distances = np.asarray(hausdorff_distances)
    hausdorff_distances_prime = np.asarray(hausdorff_distances_prime)

    hausdorff_distances /= np.max(np.abs(hausdorff_distances),axis=0)
    hausdorff_distances = np.expand_dims(hausdorff_distances, -1)
    #print(dist)
    return hausdorff_distances, hausdorff_distances_prime

def computeEDT(y):
    for i in range(y.shape[0]):
        #print(y.shape)
        y[i,:,:] = cv2.Canny(np.uint8(y[i,:,:]),0,1)
        y[i,:,:] = distance_transform_edt(np.logical_not(y[i,:,:]))
        #print(y[i,:,:])

    return y
        
        
        


# Core Function used for pyfunc
def _HausdorffGrad(op, grad,foo):
    x = op.inputs[0]
    return x, x

def _EDTGrad(op, grad):
    return 0*op.inputs[0]

def _ContourGrad(op, grad):
    return 0*op.inputs[0]   

def hausdorff_dist(y_true, y_pred):
    y_true = K.reshape(y_true, [K.tf.shape(y_true)[0],128,128])

    y_pred = K.reshape(y_pred, [K.tf.shape(y_pred)[0],128,128])
    name  = "hausdorff"
    hausdorff,_ = py_func(computeHausdorff, 
                [y_true, y_pred], 
                [tf.float32, tf.float32], 
                name = name, 
                grad=_HausdorffGrad)
    hausdorff.set_shape((None,))

    return hausdorff[0]

def computeContour(y):
    for i in range(y.shape[0]):
        y[i,:,:][y[i,:,:] < 0.5] = 0
        y[i,:,:][y[i,:,:] > 0.5] = 1
        y[i,:,:] = cv2.Canny(np.uint8(y[i,:,:]),0,1)
        #cv2.imshow("",y[i,:,:])
        #cv2.waitKey(0)
    return y
        
def chamfer_dist(y_pred, y_true):
    
    y_true = K.reshape(y_true, [K.tf.shape(y_true)[0],128,128])
    y_pred = K.reshape(y_pred, [K.tf.shape(y_pred)[0],128,128])

    y_true = py_func(computeEDT, 
                [y_true], 
                [tf.float32], 
                name = "chamfer", 
                grad=_EDTGrad)[0]

    y_pred = py_func(computeContour, 
            [y_pred], 
            [tf.float32], 
            name = "contour", 
            grad=_ContourGrad)[0]
            

    foo = K.batch_dot(y_pred, y_true)
    #foo = tf.matmul(y_pred, y_true)
    finalChamferDistanceSum = K.sum(foo, axis=0, keepdims=True) 
    #finalChamferDistanceSum = K.print_tensor(finalChamferDistanceSum, "final chamfer = ")
    finalChamferDistanceSum = tf.Print(finalChamferDistanceSum, [finalChamferDistanceSum], summarize=10)
    #finalChamferDistanceSum = K.print_tensor(finalChamferDistanceSum, "final chamfer = ")
    #print(finalChamferDistanceSum.get_shape())
    finalChamferDistanceSum = K.mean(finalChamferDistanceSum)

    #finalChamferDistanceSum = K.print_tensor(finalChamferDistanceSum, "final chamfer = ")


    return finalChamferDistanceSum



def chamfer_loss(y_true, y_pred):   
    return chamfer_dist(y_true, y_pred)

def combinedHausdorffAndDice(y_pred, y_true):
    alpha = 0.9
    beta = 1 - alpha
    dice = dice_coef_loss(y_true, y_pred)
    hd = hausdorff_dist(y_true, y_pred)
    return alpha*dice + beta*hd

def combinedHausdorffAndChamfer(y_pred, y_true):
    alpha = 0.5
    beta = 1 - alpha
    dice = dice_coef_loss(y_true, y_pred)
    cd = chamfer_dist(y_true, y_pred)
    return alpha*dice + beta*cd
     
def main():

    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d-%H:%M')
    
    num_training_patients = 1
    num_validation_patients = 1
    modes = ["flair"]
    dataHandler = SegNetDataHandler("Data/BRATS_2018/HGG", num_patients = num_training_patients, modes = modes)
    dataHandler.loadData()
    dataHandler.preprocessForNetwork()
    x_train = dataHandler.X
    x_seg_train = dataHandler.labels
    dataHandler.clear()
    
    dataHandler.setDataDirectory("Data/BRATS_2018/HGG_Validation")
    dataHandler.setNumPatients(num_validation_patients)
    dataHandler.loadData()
    dataHandler.preprocessForNetwork()
    x_val = dataHandler.X
    x_seg_val = dataHandler.labels
    dataHandler.clear()


    input_shape = (dataHandler.W,dataHandler.H, len(modes))
    
    #n_labels = x_seg_train.shape[2]
    n_labels = 1
    segnet = createSegNetWithIndexPooling(input_shape=input_shape, n_labels=n_labels, depth=2)
    lrate = 0.1
    momentum = 0.9
    #decay = lrate/num_epochs   
    sgd = SGD(lr=lrate, momentum=momentum, nesterov=True)
    segnet.compile(optimizer="adam", loss=combinedHausdorffAndChamfer, metrics=[dice_coef])

    model_directory = "Models/segnet_" + date_string 
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    log_info_filename = 'model_loss_log.csv'
    csv_logger = CSVLogger(model_directory + '/' + log_info_filename, append=True, separator=',')
    
    """
    
    image_datagen = ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    #seed = 1,
    rotation_range=3,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=False)
        

    
    
    image_datagen.fit(x_train, augment=True, seed=1)
    batch_size = 50
    segnet.fit_generator(image_datagen.flow(x_train, x_seg_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train) / batch_size,
                    validation_data=(x_val, x_seg_val),
                    epochs=50)
    
    """
    segnet.fit(x_train, x_seg_train,
                epochs=50,
                batch_size=2,
                shuffle=True,
                validation_data=(x_val, x_seg_val),
                callbacks = [csv_logger]
                )
    
    
    model_info_filename = 'model_info.txt'
    model_info_file = open(model_directory + '/' + model_info_filename, "w") 
    model_info_file.write('Number of Patients (training): ' + str(num_training_patients) + '\n')
    model_info_file.write('Number of Patients (validation): ' + str(num_validation_patients) + '\n')

    #model_info_file.write('Block Dimensions: ' + str(dataHandler.nmfComp.block_dim) + '\n')
    #model_info_file.write('Number of Components (k): ' + str(dataHandler.nmfComp.num_components) + '\n')
    model_info_file.write('\n\n')
    segnet.summary(print_fn=lambda x: model_info_file.write(x + '\n'))
    model_info_file.close();
    segnet.save(model_directory + '/model.h5')
    
    


    

if __name__ == "__main__":
   main() 
