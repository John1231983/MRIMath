'''
Created on Aug 29, 2018

@author: daniel
'''
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
from DataHandlers.UNetDataHandler import UNetDataHandler
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
from createUNet import createUNet
import tensorflow as tf
#from keras.utils.training_utils import multi_gpu_model
#import keras.backend as K
#import tensorflow as tf
from keras.optimizers import SGD
from keras.callbacks import CSVLogger

from NMFComputer.BasicNMFComputer import BasicNMFComputer
from Canny_Tensorflow import TF_Canny
import sys
import os
DATA_DIR = os.path.abspath("../")
sys.path.append(DATA_DIR)
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

def hamming_dist(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return K.mean(y_true_f*(1-y_pred_f) + y_pred_f*(1-y_true_f))

    
def diced_ham_loss(y_true, y_pred, alpha=1, beta=0.0):
    dice = dice_coef_loss(y_true, y_pred)
    ham = hamming_dist(y_true, y_pred)
    return alpha*dice + beta*ham

def computeHausdorff(A, B):
    A = tf.where(A>0.5)
    B = tf.where(B>0.5)
    
    
    #A = K.print_tensor(A, "A = ")
    #B = K.print_tensor(B, "B = ")

    A = A[:,tf.newaxis,:]
    B = B[tf.newaxis,:,:]
    
    h_ab = K.abs(A-B)
    h_ab = tf.reduce_sum(h_ab,axis = -1)
    h_ab = K.max(K.min(h_ab, axis=0))
    
    h_ba = K.abs(B-A)
    h_ba = tf.reduce_sum(h_ba,axis = -1)
    h_ba = K.max(K.min(h_ba, axis=0))
    
    return tf.maximum(h_ab, h_ba)

def hausdorff_dist(y_true, y_pred):
    
    y_true = K.reshape(y_true, [K.tf.shape(y_true)[0],128,128,1])
    y_true =  TF_Canny(y_true)
    """
    y_true = tf.map_fn(lambda x: pad_up_to(tf.where(x>0.5),[150,1],K.min(x)), y_true, dtype = tf.int64, infer_shape = False)
    y_true = K.cast(y_true, dtype = tf.float32)
    """
    
    y_pred = K.reshape(y_pred, [K.tf.shape(y_pred)[0],128,128,1])
    y_pred = TF_Canny(y_pred)
    
    res = tf.map_fn(lambda x: computeHausdorff(x[0], x[1]), (y_true,y_pred), dtype = tf.int64, infer_shape = False)
    #res = K.print_tensor(res, "Results: ")
    #print(res.get_shape())
    res.set_shape((None,))
    res = K.cast(res,dtype = tf.float32)
    res = tf.nn.l2_normalize(res)
    return res
    """
    y_pred = tf.map_fn(lambda x: pad_up_to(tf.where(x>0.5),[150,1],K.min(x)), y_pred, dtype = tf.int64, infer_shape = False)
    y_pred = K.cast(y_pred, dtype = tf.float32)
    """

    """
    y_pred = y_pred[:,tf.newaxis,:]
    y_true = y_true[tf.newaxis,:,:]
    
    h_ab = K.abs(y_pred-y_true)
    h_ab = tf.reduce_sum(h_ab,axis = -1)
    h_ab = K.print_tensor(K.min(h_ab, axis=0), message = "h_ab: ")
    h_ab = K.max(K.min(h_ab, axis=0))
    
    h_ba = K.abs(y_true-y_pred)
    h_ba = tf.reduce_sum(h_ba,axis = -1)
    h_ba = K.max(K.min(h_ba, axis=0))
    
    return tf.maximum(h_ab, h_ba)
    """
    
    #h_ab = K.print_tensor(h_ab, "distance = ")
    
    """
    diff_y_pred_y_true = K.tf.matmul(y_pred, 1/y_true)
    diff_y_pred_y_true = K.min(K.log(diff_y_pred_y_true + K.epsilon()), axis=0)

    diff_y_true_y_pred = K.tf.matmul(1/y_pred, y_true)
    diff_y_true_y_pred = K.min(K.log(diff_y_true_y_pred + K.epsilon()), axis=0)
    
    dist_a = K.max(diff_y_pred_y_true)
    dist_b = K.max(diff_y_true_y_pred)
    
    dist = K.tf.maximum(dist_a, dist_b)
    """
    #dist = K.print_tensor(dist, "dist = ")
    #dist = pairwise_l2_norm2(y_pred, y_true)

    #return h_ab
    
    



def combinedHausdorffAndDice(y_pred, y_true):
    alpha = 0.5
    beta = 1 - alpha
    dice = dice_coef_loss(y_true, y_pred)
    hd = hausdorff_dist(y_true, y_pred)
    return alpha*dice + beta*hd

def main():

    hardwareHandler = HardwareHandler()
    emailHandler = EmailHandler()
    timer = TimerModule()
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d_%H_%M')
    
    num_training_patients = 50
    num_validation_patients = 5
    
    dataHandler = UNetDataHandler("Data/BRATS_2018/HGG", BasicNMFComputer(block_dim=8), num_patients = num_training_patients, modes = ["flair"])
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
    
    dataHandler.setDataDirectory("Data/BRATS_2018/HGG_Testing")
    dataHandler.setNumPatients(1)
    dataHandler.loadData()
    dataHandler.preprocessForNetwork()
    x_test = dataHandler.X
    x_seg_test = dataHandler.labels
    dataHandler.clear()
    

    input_shape = (dataHandler.W,dataHandler.H, len(dataHandler.modes))
    
    unet = createUNet(input_shape =input_shape)
    lrate = 0.1
    momentum = 0.9
    #decay = lrate/num_epochs   
    sgd = SGD(lr=lrate, momentum=momentum, nesterov=True)
    unet.compile(optimizer="adam", loss=combinedHausdorffAndDice, metrics=[dice_coef])

    model_directory = "/home/daniel/eclipse-workspace/MRIMath/Models/unet_" + date_string
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    log_info_filename = 'model_loss_log.csv'
    csv_logger = CSVLogger(model_directory + '/' + log_info_filename, append=True, separator=',')
    
    unet.fit(x_train, x_seg_train,
                epochs=10,
                batch_size=20,
                shuffle=True,
                validation_data=(x_val, x_seg_val),
                callbacks = [csv_logger],
                )
    
    
    model_info_filename = 'model_info.txt'
    model_info_file = open(model_directory + '/' + model_info_filename, "w") 
    model_info_file.write('Number of Patients (training): ' + str(num_training_patients) + '\n')
    model_info_file.write('Number of Patients (validation): ' + str(num_validation_patients) + '\n')

    #model_info_file.write('Block Dimensions: ' + str(dataHandler.nmfComp.block_dim) + '\n')
    #model_info_file.write('Number of Components (k): ' + str(dataHandler.nmfComp.num_components) + '\n')
    model_info_file.write('\n\n')
    unet.summary(print_fn=lambda x: model_info_file.write(x + '\n'))
    model_info_file.close();
    unet.save(model_directory + '/model.h5')
    
    decoded_imgs = unet.predict(x_test)
    
    n = 100
    for i in range(n):
        fig = plt.figure()
        plt.gray();   
        a=fig.add_subplot(1,3,1)
        plt.imshow(x_test[i,:,:,0])
        plt.axis('off')
        plt.title('Original')
        
        a=fig.add_subplot(1,3,2)
        plt.imshow(x_seg_test[i,:,:,0])
        plt.axis('off')
        plt.title('GT Segment')
        
        a=fig.add_subplot(1,3,3)
        plt.imshow(decoded_imgs[i,:,:,0])
        plt.axis('off')
        plt.title('Predicted Segment')

        plt.show()
    
    

if __name__ == "__main__":
   main() 
