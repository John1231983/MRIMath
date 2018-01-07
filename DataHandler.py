'''
Created on Jan 1, 2018

@author: daniel
'''

import os
import cv2
import numpy as np
import multiprocessing
from joblib import Parallel, delayed


class DataHandler:
    
    W = 240
    H = 240
    
    mri_images = []
    mri_segment_data = [[] for _ in range(8)]



    def getImage(self, path):
        path=path.decode()
        img = cv2.imread(path,0)
        return img

    def loadDataParallel(self, training_directory, start, finish):
        X_train = []
        segment_data = [[] for _ in range(8)]
        print('Reading images')
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(self.loadIndividualImage(training_directory, i, X_train, segment_data) for i in range(start, finish)))
    
    def loadIndividualImage(self, training_directory, j, mri_images, mri_segment_data):
        if((j>0 and j < 107) or j > 135):
            print('Reading Patient ' + str(j))
            if j < 10:
                directory = os.fsencode(training_directory + '/Patient_(00' + str(j)  + ')_data/')
            elif j < 100:
                directory = os.fsencode(training_directory + '/Patient_(0' + str(j)  + ')_data/')
            else:
                directory = os.fsencode(training_directory + '/Patient_(' + str(j)  + ')_data/')
            for file in os.listdir(directory + b'/Original_Img_Data'):
                img = self.getImage(directory+b'/Original_Img_Data/'+file)
                mri_images.append(img)
            segment_directory = os.fsencode(directory + b'Segmented_Img_Data')
            for dir in os.listdir(segment_directory):
                for file in os.listdir(segment_directory+b'/'+dir):
                    ind = file[4:5]
                    mri_segment_data[int(ind.decode())-1].append(self.getImage(segment_directory+b'/'+dir+b'/'+file))
        
    def getMRIData(self):
        print(len(self.mri_images))
        return self.preprocessForNetwork(self.mri_images, self.mri_segment_data)
    
    def clearMRIData(self):
        self.mri_images = []
        self.mri_segment_data = [[] for _ in range(8)]

    def preprocessForNetwork(self, training_data, segment_data):
        n_imgs = len(training_data)
        training = np.array(training_data)
        training = training.reshape(n_imgs,self.W,self.H,1)
        training = training.astype('float32') / 255;
        segments = np.array(segment_data);
        segments = segments.reshape(n_imgs,self.W,self.H,8)
        return training, segments
    

        
        