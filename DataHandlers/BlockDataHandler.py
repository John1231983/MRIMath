'''
Created on Aug 1, 2018

@author: daniel
'''
from DataHandlers.DataHandler import DataHandler
import numpy as np
from keras.utils import np_utils
from scipy.stats import mode
import matplotlib.pyplot as plt
import os
import nibabel as nib
from Utils.TimerModule import TimerModule
import cv2 
from imblearn.over_sampling import SMOTE
from collections import Counter
from multiprocessing import Pool
from functools import partial
class BlockDataHandler(DataHandler):
    
    nmfComp = None
    def __init__(self, dataDirectory, nmfComp, W = 240, H = 240, num_patients = 3, load_mode = "training" ):
        super().__init__(dataDirectory, W, H, num_patients, load_mode)
        self.nmfComp = nmfComp
        
        
    def processData(self, image, seg_image, ind):
        X = []
        y = []
        
        image = image[:,:,ind]
        seg_image = seg_image[:,:,ind]

        rmin,rmax, cmin, cmax = self.bbox(image)
        
        image = image[rmin:rmax, cmin:cmax]
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR_EXACT)

        seg_image = seg_image[rmin:rmax, cmin:cmax]
        seg_image = cv2.resize(seg_image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR_EXACT)
        #seg_image[seg_image > 0] = 1
        
        W, H = self.nmfComp.run(image)
        seg_image[seg_image > 0] = 1
        labels = self.getLabels(seg_image)
        
        H = np.nan_to_num(H)
        H_cols = np.hsplit(H, H.shape[1])
        
        
        max_num_background_blocks = np.count_nonzero(seg_image)
        num_background_blocks = 0
        for i, label in enumerate(labels):
            
            if label == 0:
                num_background_blocks = num_background_blocks + 1
                if num_background_blocks > max_num_background_blocks:
                    continue
            X.append(H_cols[i])
            y.append(label)


                   
        return X, y
     
    def loadData(self, mode):
        main_dir = os.listdir(self.dataDirectory)[0:self.num_patients+1]
        for subdir in main_dir:
            image_dir = self.dataDirectory + "/" + subdir
            data_dirs = os.listdir(image_dir)
            seg_image = nib.load(image_dir+
                                   "/" + 
                                   [s for s in data_dirs if "seg" in s][0]).get_fdata(caching = "unchanged",
                                                                                      dtype = np.float32)
            
            pool = Pool(processes=6)
            inds = [i for i in list(range(155)) if np.count_nonzero(seg_image[:,:,i]) > 0]
            for path in os.listdir(self.dataDirectory + "/" + subdir):
                if mode + ".nii" in path:
                    image = nib.load(image_dir + "/" + path).get_fdata(caching="unchanged", dtype = np.float32)
                    temp = [self.processData(image, seg_image, i) for i in inds]
                    #temp = pool.map(partial(self.processData, image, seg_image), inds)
                    foo, bar = zip(*temp)
                    self.X.extend([item for sublist in foo for item in sublist])
                    self.labels.extend([item for sublist in bar for item in sublist])
                    break

    def getLabels(self, seg_image):
        m = self.nmfComp.block_dim
        cols = np.vsplit(seg_image, seg_image.shape[0]/m)
        row_split = [np.hsplit(c,seg_image.shape[0]/m) for c in cols]
        labels = [mode(block[0])[0] for sublist in row_split for block in sublist]
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
        self.labels = np.array(self.labels)

        
