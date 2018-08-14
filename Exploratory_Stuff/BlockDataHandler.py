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
        max_background_blocks = int(0.1*H.shape[1])
        max_2_blocks = int(0.2*H.shape[1])
        num_background_blocks = 0
        num_2_blocks = 0

        #seg_image[seg_image > 0] = 1
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
            elif block == 2:
                num_2_blocks = num_2_blocks +1
                if num_2_blocks > max_2_blocks:
                    continue
            y.append(block)
            X.append(H_cols[i])
                        
        return X, y
    
    def performNMFOnSlice(self, image, seg_image, i):
        #image[:,:,i] = self.preprocess(image[:,:,i])        
        W, H = self.nmfComp.run(image[:,:,i])
        return self.processData(W,H, seg_image[:,:,i])
    
    def preprocessForNetwork(self):
        n_imgs = len(self.X)
        
        self.X = np.array( self.X )
        self.X = self.X.reshape(n_imgs,self.nmfComp.num_components)
        #self.labels = np.array( self.labels )
        self.labels = np_utils.to_categorical(self.labels)
        #self.labels = self.labels.reshape(n_imgs,self.W,self.H,1)
        # self.labels = self.labels.reshape(n_imgs, self.W*self.H,2)
