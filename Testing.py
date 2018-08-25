'''
Created on Aug 9, 2018

@author: daniel
'''
import sys
import os
from numpy import genfromtxt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.TimerModule import TimerModule
from Exploratory_Stuff.BlockDataHandler import BlockDataHandler
#from keras.callbacks import CSVLogger,ReduceLROnPlateau
from keras.layers import Dense, BatchNormalization, Conv1D
from keras.models import Sequential
from keras.callbacks import CSVLogger
#from keras.optimizers import SGD
#import os
from keras.models import load_model

from Utils.EmailHandler import EmailHandler
from Utils.HardwareHandler import HardwareHandler
from datetime import datetime
#from keras.utils.training_utils import multi_gpu_model
#import keras.backend as K
#import tensorflow as tf
import nibabel as nib
import numpy as np
from NMFComputer.BasicNMFComputer import BasicNMFComputer
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import LeakyReLU, PReLU
import matplotlib.patches as patches

DATA_DIR = os.path.abspath("../")
sys.path.append(DATA_DIR)
def main():
    block_dim=8
    num_components=8
    nmfComp = BasicNMFComputer(block_dim=block_dim, num_components=num_components)
    dataHandler = BlockDataHandler("Data/BRATS_2018/HGG", nmfComp, num_patients = 0)
    mode = "flair"
    #model = load_model("Models/2018-08-13_17_59_t1ce/model.h5")
    test_data_dir = "Data/BRATS_2018/HGG"
    image = None
    seg_image = None
    for subdir in os.listdir(test_data_dir):
        for path in os.listdir(test_data_dir+ "/" + subdir):
            if mode in path:
                image = nib.load(test_data_dir + "/" + subdir + "/" + path).get_data()
                seg_image = nib.load(test_data_dir+ "/" + subdir + "/" + path.replace(mode, "seg")).get_data()
                break
        break
    
    inds = [i for i in list(range(155)) if np.count_nonzero(seg_image[:,:,i]) > 0.01*seg_image[:,:,i].size]
    
    for k in inds:
        #image[:,:,k] = dataHandler.preprocess(image[:,:,k])
        W, H = nmfComp.run(image[:,:,k])
        H_cols = np.hsplit(H, H.shape[1])
        W_cols = np.hsplit(W, W.shape[1])
        
        rows = np.vsplit(image[:,:,k], image[:,:,k].shape[0]/nmfComp.block_dim)
        col_split = [np.hsplit(r,image[:,:,k].shape[0]/nmfComp.block_dim) for r in rows]
        blocks = [item for sublist in col_split for item in sublist]
        m = len(blocks)
        for i in range(len(blocks)):
            if np.count_nonzero(blocks[i]) <= 0:
                continue
            
            fig = plt.figure()
            
            ax=fig.add_subplot(1,4,1)
            plt.gray()
            plt.imshow(image[:,:,k])
            #rect = patches.Rectangle((block_dim*count, j),nmfComp.block_dim,nmfComp.block_dim,linewidth=1,edgecolor='r',facecolor='none')
            #ax.add_patch(rect)
            plt.axis('off')
            plt.title('Original Image')



            fig.add_subplot(1,4,2)
            
            plt.gray()
            plt.imshow(blocks[i])
            plt.axis('off')
            plt.title('Block')
            
            fig.add_subplot(1,4,3)

            plt.bar(list(range(nmfComp.num_components)), H_cols[i].T.tolist()[0], align='center')
            plt.ylabel('Regions')
            plt.title('Regional Distribution')
            
            fig.add_subplot(1,4,4)
            for n in range(nmfComp.num_components):
                #plt.hist(W_cols[ind].T.tolist()[0], density=True, bins=nmfComp.num_hist_bins)
                plt.bar(list(range(nmfComp.num_hist_bins)), W_cols[n].T.tolist()[0], label='Region ' + str(n))
                plt.xlabel('Grayscale Value')
                plt.title('Grayscale Regional Distribution')
                plt.legend()
            dataHandler.showRegions(W, H, image[:,:,k], seg_image[:,:,k])
            plt.show()

                
                #seg_est[j:j+m, i:i+m] = np.full((m, m), np.argmax(labels[ind]))
        """
        fig = plt.figure()
        plt.gray();
        a=fig.add_subplot(1,3,1)
        plt.imshow(image[:,:,k])
        plt.axis('off')
        plt.title('Original')

        a=fig.add_subplot(1,3,2)
        plt.imshow(seg_image[:,:,k])
        plt.axis('off')
        plt.title('GT Segment')
        
        a=fig.add_subplot(1,3,3)
        plt.imshow(seg_est)
        plt.axis('off')
        plt.title('Estimate Segment')
        plt.show()
        """
if __name__ == "__main__":
   main() 