'''
Created on Jan 1, 2018

@author: daniel
'''

import os
import cv2
import numpy as np
from functools import partial
from HardwareHandler import HardwareHandler
from math import floor
import threading
import matplotlib.pyplot as plt


class DataHandler:
    
    lock = threading.Lock()
    X = []
    labels = []
    W = 240
    H = 240
    hardwareHandler = HardwareHandler()

    def getImage(self, path):
        path=path.decode()
        img = cv2.imread(path,0)
        return img

    def loadDataSequential(self, training_directory, start, finish):
        X_train = []
        segment_data = [[] for _ in range(8)]
        print('Reading images')
        for j in range(start,finish):
            self.loadIndividualImage(j, training_directory, X_train, segment_data)
        training, segments = self.preprocessForNetwork(X_train, segment_data)
        return training, segments
    
    def loadDataParallel(self, data_directory, start, finish):
        segment_data = [[] for _ in range(8)]
        print('Reading images')
        pool = self.hardwareHandler.createThreadPool()
        pool.map(partial(self.loadIndividualImage, data_directory=data_directory), range(start, finish))
        #training, segments = self.preprocessForNetwork(X_train, segment_data)
        pool.terminate()
        #return training, segments
        
    def loadIndividualImage(self, index, data_directory):
        print('Reading Patient ' + str(index))
        img_directory = os.fsencode(self.getDirectoryFromIndex(index, data_directory))
        for file in os.listdir(img_directory + b'/Original_Img_Data'):
            img = self.getImage(img_directory+b'/Original_Img_Data/'+file)
            length, width = img.shape
            if(length != 240 or width != 240):
                print('Could not add patient  ' + str(index) + ' because dimensions did not match')
                print('Width: ' + str(width) + ", Length: " + str(length))
            else:
                self.X.append(img)
                
    def getDirectoryFromIndex(self, index, data_directory):
        if index < 10:
            patient_directory = data_directory + '/Patient_(00' + str(index)  + ')_data/'
        elif index < 100:
            patient_directory = data_directory + '/Patient_(0' + str(index)  + ')_data/'
        else:
            patient_directory = data_directory + '/Patient_(' + str(index)  + ')_data/'
        return patient_directory
        
    def derivePatches(self, data_directory, n):
        print(len(self.X))
        for ind in range(1,len(self.X)):
            label = dict()
            for m in range(0, 8):
                label[m] = 0
            segment_directory = os.fsencode(self.getDirectoryFromIndex(ind, data_directory)) + b'/Segmented_Img_Data'
            for dir in os.listdir(segment_directory):
                for file in os.listdir(segment_directory+b'/'+dir):
                    seg_img = self.getImage(segment_directory+b'/'+dir+b'/'+file)
                    seg_num = file[4:5]
                    length, width = self.X[ind].shape
                    for col in range(1,width-n):
                        for row in range(1,length-n):
                            if(col+n >= length):
                                col = 0
                                row = row + 1
                            #if(row+n >= width):
                                
                                
                            patch = seg_img[row:row+n,col:col+n]
                            if(patch[floor(n/2),floor(n/2)] == 255):
                                label[seg_num] = 1
                    self.labels.append(label)
                    print(len(self.labels))

                                
                                
        
    def preprocessForNetwork(self, training_data, segment_data):
        n_imgs = len(training_data)
        training = np.array(training_data)
        training = training.reshape(n_imgs,self.W,self.H,1)
        training = training.astype('float32') / 255;
        segments = np.array(segment_data);
        segments = segments.reshape(n_imgs,self.W,self.H,8)
        return training, segments
    

        
        