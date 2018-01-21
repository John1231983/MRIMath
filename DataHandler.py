'''
Created on Jan 1, 2018

@author: daniel
'''

import os
import cv2
from functools import partial
from HardwareHandler import HardwareHandler
import threading
from random import randint
import numpy as np
from math import floor
from multiprocessing import Process, Manager
from keras.utils import np_utils

#import random


class DataHandler:
    
    lock = threading.Lock()
    manager = Manager()
    X = []
    labels = []
    W = 240
    H = 240
    hardwareHandler = HardwareHandler()
    
    tolerance = None #the percentage of background pixels permitted in a patch
    numPatches = None #the number of patches to extract from an imaage
    n = None #dimensions of each patch (n x n)
    #stepSize = None # the stride, in the case of a sliding window approach
    def __init__(self, tolerance = 0.25, numPatches = 5, n = 25):
        self.tolerance = tolerance
        self.numPatches = numPatches
        self.n = n
    
    def getImage(self, path):
        path=path.decode()
        img = cv2.imread(path,0)
        return img

    def loadDataSequential(self, training_directory, start, finish):
        print('Reading images')
        for j in range(start,finish):
            self.loadIndividualImage(j, training_directory)

    
    def loadDataParallel(self, data_directory, start, finish):
        print('Reading images')
        self.X = self.manager.list()  
        self.labels = self.manager.list()  
        processes = []
        for i in range(start, finish):
            p = Process(target=self.loadIndividualImage, args=(i, data_directory))  
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        #pool = self.hardwareHandler.createThreadPool()
        #pool.map(partial(self.loadIndividualImage, data_directory=data_directory), range(start, finish))
        #pool.terminate()
    
        
    def loadIndividualImage(self, index, data_directory):
        print('Reading Patient ' + str(index))
        patient_directory = os.fsencode(self.getDirectoryFromIndex(index, data_directory))
        data_dir = patient_directory + b'/Original_Img_Data'
        for file in os.listdir(data_dir):
            img = self.getImage(data_dir+b'/'+file)
            length, width = img.shape
            if(length != self.H or width != self.W):
                print('Could not add patient  ' + str(index) + ' because dimensions did not match')
                print('Width: ' + str(width) + ", Length: " + str(length))
            else:
                for _ in range(0,self.numPatches):
                    self.deriveRandomPatch(patient_directory,img, file)

    def getDirectoryFromIndex(self, index, data_directory):
        if index < 10:
            patient_directory = data_directory + '/Patient_(00' + str(index)  + ')_data/'
        elif index < 100:
            patient_directory = data_directory + '/Patient_(0' + str(index)  + ')_data/'
        else:
            patient_directory = data_directory + '/Patient_(' + str(index)  + ')_data/'
        return patient_directory
                
                            
           
    def preprocessForNetwork(self):
        n_imgs = len(self.X)
        self.X = np.array(self.X)
        self.X = self.X.reshape(n_imgs,self.n,self.n,1)
        self.X = self.X.astype('float32') / 255;
        self.labels = np.array(self.labels);
        np_utils.to_categorical(self.labels, nb_classes=8)
        #self.labels = np.array(self.labels);
        #self.labels = self.labels.reshape(n_imgs,8)
    
    def derivePatches(self, img, stepSize):
        for (x, y, window) in self.deriveIndividualPatch(img, stepSize):
            if window.shape[0] != self.n or window.shape[1] != self.n:
                continue
            #self.X.append(window)
        
    def deriveRandomPatch(self, patient_directory,img, file):
        self.lock.acquire()
        x = min(randint(1, self.W), self.W - self.n)
        y = min(randint(1, self.H), self.H - self.n)
        patch = img[x:x+self.n, y:y+self.n]
        #numBackgroundPixels = np.sum(patch == 0)
        #if numBackgroundPixels > self.tolerance*img.size:
        #    return self.deriveRandomPatch(patient_directory,img,file)
        #else:
        self.X.append(patch)
        self.derivePatchFromSegments(patient_directory, x,y, file)
        self.lock.release()

                   
            
    def deriveIndividualPatch(self, img, stepSize, n):
    # slide a window across the image
        for y in range(0, img.shape[0], stepSize):
            for x in range(0, img.shape[1], stepSize):
            # yield the current window
                yield (x, y, img[y:y + n, x:x + n])
                
    def derivePatchFromSegments(self, patient_dir, x ,y, img_num):
#             label = []
#             for _ in range(0, 8):
#                 label.append(0)
            label = 0
            segment_directory = os.fsencode(patient_dir) + b'/Segmented_Img_Data'
            #for dir in os.listdir(segment_directory):
            for file in os.listdir(segment_directory+b'/'+img_num[0:len(img_num)-4]):
                seg_img = self.getImage(segment_directory+b'/'+img_num[0:len(img_num)-4]+b'/'+file)
                seg_num = int(file[4:5].decode("utf-8"))
                seg_patch = seg_img[x:x+self.n, y:y+self.n]
                if(seg_patch[floor(self.n/2),floor(self.n/2)] == 255):
                    label = seg_num
                    
                    
            self.labels.append(label)
    
    def getData(self):
        self.preprocessForNetwork()
        return self.X, self.labels
    
    def clearVectors(self):
        self.X =[]
        self.labels = []

        
        
