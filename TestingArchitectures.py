'''
Created on Jul 10, 2018

@author: daniel
'''

#from multiprocessing import Process, Manager
#from keras.utils import np_utils
import sys
import os
from keras.utils import np_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from DataHandlers.SegNetDataHandler import SegNetDataHandler
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from Mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D
import math
from CustomLosses import dice_coef, dice_coef_multilabel, dice_coef_loss, combinedDiceAndChamfer
from dipy.segment.mask import clean_cc_mask

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

    
    num_testing_patients = 1
    n_labels = 1
    normalize = True
    modes = ["flair"]
    dataHandler = SegNetDataHandler("Data/BRATS_2018/HGG_Testing", 
                                    num_patients = num_testing_patients, 
                                    modes = modes)
    dataHandler.loadData()
    dataHandler.preprocessForNetwork()
    x_test = np.array(dataHandler.X)
    x_seg_test = dataHandler.labels
    dataHandler.clear()

    segnet = load_model("Models/segnet_2018-10-25-23:27/model.h5", custom_objects={'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D, 
                                                                               'MaxUnpooling2D':MaxUnpooling2D, 
                                                                               'combinedDiceAndChamfer':combinedDiceAndChamfer, 
                                                                               'dice_coef':dice_coef, 
                                                                               'dice_coef_loss':dice_coef_loss,
                                                                           'dice_coef_multilabel' : dice_coef_multilabel})
    
    
    
    if normalize:
        mu = np.mean(x_test)
        sigma = np.std(x_test)
        x_test -= mu
        x_test /= sigma
    decoded_imgs = segnet.predict(x_test)

    if n_labels > 1:
        #x_seg_test = np_utils.to_categorical(x_seg_test)
        #x_seg_test = np.argmax(x_seg_test, axis=3)

        decoded_imgs = np.argmax(decoded_imgs, axis=2)
        print(x_seg_test.shape)
    else:
        for x in x_seg_test:
            x[x > 0.5] = 1
            x[x < 0.5] = 0
        

    



    avg_dice = 0
    N = len(decoded_imgs)
    """
    for i in range(N):
            decoded_imgs[i][decoded_imgs[i] < 0.5] = 0
            dice = computeDice(x_seg_test[i], np.squeeze(decoded_imgs[i]))
            avg_dice = avg_dice + dice
    print(str(avg_dice/N))
    """
    for i in range(N):
        fig = plt.figure()
        plt.gray();   
        fig.add_subplot(1,3,1)
        plt.imshow(x_test[i,:,:,0])
        plt.axis('off')
        plt.title('Original')
        
        fig.add_subplot(1,3,2)
        plt.imshow(x_seg_test[i])
        plt.axis('off')
        plt.title('GT Segment')
        
        fig.add_subplot(1,3,3)

        plt.imshow(decoded_imgs[i].reshape(dataHandler.W, dataHandler.W))
        plt.axis('off')
        plt.title('Predicted Segment')

        plt.show()
    


if __name__ == "__main__":
   main() 
