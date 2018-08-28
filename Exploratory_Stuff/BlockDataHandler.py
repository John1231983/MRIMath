'''
Created on Aug 1, 2018

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

class BlockDataHandler(DataHandler):
    
    def __init__(self, dataDirectory, nmfComp, W = 128, H = 128, num_patients = 3):
        super().__init__(dataDirectory, nmfComp, W, H, num_patients)
        
        
    def processData(self, image, seg_image):
        X = []
        y = []
        
        rmin,rmax, cmin, cmax = self.bbox(image)
        
        image = image[rmin:rmax, cmin:cmax]
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR_EXACT)

        seg_image = seg_image[rmin:rmax, cmin:cmax]
        seg_image = cv2.resize(seg_image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR_EXACT)
        
        W, H = self.nmfComp.run(image)
        labels = self.getLabels(seg_image)
        
        H = np.nan_to_num(H)
        H_cols = np.hsplit(H, H.shape[1])
        
        
        max_num_background_blocks = np.count_nonzero(labels)
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
        timer = TimerModule()
        timer.startTimer()
        J = 0
        for subdir in os.listdir(self.dataDirectory):
            if J > self.num_patients:
                timer.stopTimer()
                print(timer.getElapsedTime())
                break
            for path in os.listdir(self.dataDirectory + "/" + subdir):
                if mode in path:
                    image = nib.load(self.dataDirectory + "/" + subdir + "/" + path).get_data()
                    seg_image = nib.load(self.dataDirectory + "/" + subdir + "/" + path.replace(mode, "seg")).get_data()
                    inds = [i for i in list(range(155)) if np.count_nonzero(seg_image[:,:,i]) > 0]
                    if inds is []:
                        continue
                    temp = [self.processData(image[:,:,i], seg_image[:,:,i]) for i in inds]
                    foo = [i[0] for i in temp]
                    bar = [i[1] for i in temp]
                    self.X.extend([item for sublist in foo for item in sublist])
                    self.labels.extend([item for sublist in bar for item in sublist])
                    """
                    temp = [self.performNMFOnSlice(image, seg_image, i) for i in inds]
                    with Pool(processes=8) as pool:
                        temp = pool.map(partial(self.performNMFOnSlice, image, seg_image), inds)
                    foo = [i[0] for i in temp]
                    bar = [i[1] for i in temp]

                    self.X.extend([item for sublist in foo for item in sublist])
                    self.labels.extend([item for sublist in bar for item in sublist])
                    """
                    J = J + 1
                    break

    def getLabels(self, seg_image):
        m = self.nmfComp.block_dim
        cols = np.vsplit(seg_image, seg_image.shape[0]/m)
        row_split = [np.hsplit(c,seg_image.shape[0]/m) for c in cols]
        labels = [block[0][int(m/2)] for sublist in row_split for block in sublist]
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
