'''
Created on Aug 1, 2018

@author: daniel
'''
from Exploratory_Stuff.DataHandler import DataHandler
import numpy as np
from keras.utils import np_utils
from scipy.stats import mode

class BlockDataHandler(DataHandler):
    
    def __init__(self, dataDirectory, nmfComp, W = 240, H = 240, num_patients = 3):
        super().__init__(dataDirectory, nmfComp, W, H, num_patients)
        
        
    def processData(self, W, H, seg_image):
        X = []
        y = []
        m = self.nmfComp.block_dim
        max_background_blocks = 0.01*H.shape[1]
        num_background_blocks = 0
        seg_image[seg_image > 0] = 1
        H = np.nan_to_num(H)
        cols = np.hsplit(seg_image, seg_image.shape[0]/m)
        row_split = [np.vsplit(c,seg_image.shape[0]/m) for c in cols]
        blocks = [int(mode(block, axis=None)[0][0]) for sublist in row_split for block in sublist]
        H_cols = np.hsplit(H, H.shape[1])
        
        for i, block in enumerate(blocks):
            if block == 0:
                num_background_blocks = num_background_blocks + 1
                if num_background_blocks > max_background_blocks:
                    continue
            y.append(block)
            X.append(H_cols[i])
                        
        return X, y
    
    def preprocessForNetwork(self):
        n_imgs = len(self.X)
        
        self.X = np.array( self.X )
        self.X = self.X.reshape(n_imgs,self.nmfComp.num_components)
        #self.labels = np.array( self.labels )
        self.labels = np_utils.to_categorical(self.labels)
        #self.labels = self.labels.reshape(n_imgs,self.W,self.H,1)
        # self.labels = self.labels.reshape(n_imgs, self.W*self.H,2)