'''
Created on Aug 1, 2018

@author: daniel
'''
from Exploratory_Stuff.DataHandler import DataHandler
import numpy as np
from keras.utils import np_utils
from scipy.stats import mode
import matplotlib.pyplot as plt

class BlockDataHandler(DataHandler):
    
    def __init__(self, dataDirectory, nmfComp, W = 240, H = 240, num_patients = 3):
        super().__init__(dataDirectory, nmfComp, W, H, num_patients)
        
        
    def processData(self, W, H, seg_image):
        X = []
        y = []
        
        max_background_blocks = int(0.1*H.shape[1])
        num_background_blocks = 0
        H = np.nan_to_num(H)
        
        labels = self.getLabels(seg_image)
        H_cols = np.hsplit(H, H.shape[1])
        for i, label in enumerate(labels):
            if label == 0:
                num_background_blocks = num_background_blocks + 1
                if num_background_blocks > max_background_blocks:
                    continue
            y.append(label)
            X.append(H_cols[i])
                        
        return X, y
     
    def getLabels(self, seg_image):
        m = self.nmfComp.block_dim
        cols = np.vsplit(seg_image, seg_image.shape[0]/m)
        row_split = [np.hsplit(c,seg_image.shape[0]/m) for c in cols]
        labels = [np.argmax(np.bincount(block[0])) for sublist in row_split for block in sublist]
        return labels
    def showRegions(self, W, H, image, seg_image):
        regions = np.argmax(H, axis=0)
        N = H.shape[0]
        m = self.nmfComp.block_dim
        fig = plt.figure()
        plt.gray();
        fig.add_subplot(1,N+1,1)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Original')
        
        for i in range(1,N+1):
            region = regions.copy();
            region_image = np.zeros((240,240))
            region[regions > i] = 0
            region[regions < i] = 0
            #region = region.astype(bool)
            fig.add_subplot(1,N+2,i+1)
            ind = 0
            for j in range(0, seg_image.shape[0], m):
                for k in range(0, seg_image.shape[1], m):
                    region_image[j:j+m, k:k+m] = region[ind]
                    ind = ind+1
            plt.imshow(region_image)
            plt.axis('off')
            plt.title(str(i-1))
        fig.add_subplot(1,N+2,N+2)
        plt.imshow(seg_image)
        plt.axis('off')
        plt.title('Segment')
        #plt.show()
    def performNMFOnSlice(self, image, seg_image, i):
        #image[:,:,i] = self.preprocess(image[:,:,i])        
        W, H = self.nmfComp.run(image[:,:,i])
        #self.showRegions(W, H, image[:,:,i], seg_image[:,:,i])
        return self.processData(W,H, seg_image[:,:,i])
    
    def preprocessForNetwork(self):
        n_imgs = len(self.X)
        
        self.X = np.array( self.X )
        self.X = self.X.reshape(n_imgs,self.nmfComp.num_components)
        #self.labels = np.array( self.labels )
        self.labels = np_utils.to_categorical(self.labels)
        #self.labels = self.labels.reshape(n_imgs,self.W,self.H,1)
        # self.labels = self.labels.reshape(n_imgs, self.W*self.H,2)
