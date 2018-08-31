'''
Created on Aug 29, 2018

@author: daniel
'''
from Exploratory_Stuff.DataHandler import DataHandler
import numpy as np
from keras.utils import np_utils
from scipy.stats import mode
import matplotlib.pyplot as plt
import os
import nibabel as nib
from Utils.TimerModule import TimerModule
import cv2 
from multiprocessing import Pool
from functools import partial
class ExtendedBlockDataHandler(DataHandler):
    modes = ["flair", "t1ce", "t1", "t2"]
    def __init__(self, dataDirectory, nmfComp, W = 240, H = 240, num_patients = 3):
        super().__init__(dataDirectory, nmfComp, W, H, num_patients)
        
        
    def processData(self, image, label_indices):
        X = []
        """
        rmin,rmax, cmin, cmax = self.bbox(image)
        
        image = image[rmin:rmax, cmin:cmax]
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR_EXACT)

        seg_image = seg_image[rmin:rmax, cmin:cmax]
        seg_image = cv2.resize(seg_image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR_EXACT)
        #seg_image[seg_image > 0] = 1
        """
        _, H = self.nmfComp.run(image)
        H = np.nan_to_num(H)
        H_cols = np.hsplit(H, H.shape[1])
        
        
        for i in label_indices:
            X.append(H_cols[i])
                   
        return X
     
    def processData2(self, image):
        _, H = self.nmfComp.run(image)
        H = np.nan_to_num(H)
        H_cols = np.hsplit(H, H.shape[1])   
        return H_cols
    
        
    def loadData(self):
        J = 0
        for subdir in os.listdir(self.dataDirectory):
            if J > self.num_patients:
                break
            data_dirs = os.listdir(self.dataDirectory + "/" + subdir)
            seg_image = nib.load(self.dataDirectory + "/" + subdir + "/" + [s for s in data_dirs if "seg" in s][0]).get_data()
            inds = [i for i in list(range(90)) if np.count_nonzero(seg_image[:,:,i]) > 0]
            valid_label_indices = {}
            for i in inds:
                valid_label_indices[i] = self.getLabels(seg_image[:,:,i]) 
            foo = []

            for path in data_dirs:
                for mode in self.modes:
                    if mode in path:
                        image = nib.load(self.dataDirectory + "/" + subdir + "/" + path).get_data()
                        temp = [self.processData(image[:,:,i], valid_label_indices[i]) for i in inds]
                        bar = [i for i in temp]
                        foo.extend([item for sublist in bar for item in sublist])
                        break;
            J = J + 1
            chunks = [foo[x:x+int(len(foo)/len(self.modes))] for x in range(0, len(foo), int(len(foo)/len(self.modes)))]
            for i in range((int(len(foo)/len(self.modes)))):
                self.X.append(np.concatenate((chunks[0][i], chunks[1][i], chunks[2][i],chunks[3][i]), axis=None))
                



    def getLabels(self, seg_image):
        m = self.nmfComp.block_dim
        cols = np.vsplit(seg_image, seg_image.shape[0]/m)
        row_split = [np.hsplit(c,seg_image.shape[0]/m) for c in cols]
        labels = [mode(block[0])[0] for sublist in row_split for block in sublist]
        num_background_blocks = 0
        max_num_background_blocks = 0.01*len(labels)
        inds = []
        for i, label in enumerate(labels):
            if label == 0:
                num_background_blocks = num_background_blocks + 1
                if num_background_blocks > max_num_background_blocks:
                    continue
            self.labels.append(label)
            inds.append(i)
        return inds
    
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
        self.X = self.X.reshape(n_imgs,len(self.modes)*self.nmfComp.num_components)
        #self.labels = np.array( self.labels )
        self.labels = np_utils.to_categorical(self.labels)
        #self.labels = self.labels.reshape(n_imgs,self.W,self.H,1)
        # self.labels = self.labels.reshape(n_imgs, self.W*self.H,2)
