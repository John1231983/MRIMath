'''

Class designed to do all data handling and manipulation, ranging from dataloading to network preprocessing.
As time goes on, some of this may be refactored, and some of this functionality is contingent on the data being stored 
in a certain structure. 

@author Daniel Enrico Cahall

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
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(10000)

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
    
    


    ## The constructor for the datahandler class
    #
    # @param tolerance the percentage of pixels in a patch that can be background (default 0.25)
    # @param numPatches the number of patches to extract per image (default 10)
    # @param n the dimensions of the patch to be taken from the image (default 25)
    def __init__(self, tolerance = 0.1, numPatches = 10, n = 25):
        self.tolerance = tolerance
        self.numPatches = numPatches
        self.n = n
    
    
    ## Reads an image from a filepath
    #
    # @param path the path to an image file
    # @return the image from the filepath (if one existed) as a numpy array
    def getImage(self, path):
        path=path.decode()
        img = cv2.imread(path,0)
        return img


    ## Loads patient images and segments sequentially, assuming you want to through a range of numbered patients
    #
    # @param data_directory the directory where all patient data is located
    # @param start the patient number to start with (inclusive)
    # @param finish the patient number to stop at (exclusive)
    def loadDataSequential(self, data_directory, start, finish):
        print('Reading images')
        for j in range(start,finish):
            self.loadIndividualPatient2(j, data_directory)

    ## Loads patient images and segments in parallel, assuming you want to through a range of numbered patients
    #
    # @param data_directory the directory where all patient data is located
    # @param start the patient number to start with (inclusive)
    # @param finish the patient number to stop at (exclusive)
    def loadDataParallel(self, data_directory, start, finish):
        print('Reading images')
        self.X = self.manager.list()  
        self.labels = self.manager.list()  
        processes = []
        for i in range(start, finish):
            p = Process(target=self.loadIndividualPatient2, args=(i, data_directory))  
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    
    ## Derives and labels patches from an individual patient
    #
    # @param data_directory the directory where all patient data is located
    # @param index index of the patient in the numbered directory
    def loadIndividualPatient(self, index, data_directory):
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
                    self.deriveRandomPatch(patient_directory, img, file)

    ## TBD
    #
    # @param data_directory the directory where all patient data is located
    # @param index index of the patient in the numbered directory
    def loadIndividualPatient2(self, index, data_directory):
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
                self.deriveRegionsOfInterest(patient_directory,data_directory,img, file)
    ## Constructs the patient directory string based on the index (based on current labeling scheme)
    #
    # @param index the index of the patient that you need the specific directory for
    # @param data_directory Directory where all patient data is located
    def getDirectoryFromIndex(self, index, data_directory):
        if index < 10:
            patient_directory = data_directory + '/Patient_(00' + str(index)  + ')_data/'
        elif index < 100:
            patient_directory = data_directory + '/Patient_(0' + str(index)  + ')_data/'
        else:
            patient_directory = data_directory + '/Patient_(' + str(index)  + ')_data/'
        return patient_directory
                
    ## Preprocesses the data for the network by converting the list of patches and labels to a numpy array, and
    # normalizing the patches               
    def preprocessForNetwork(self):
        n_imgs = len(self.X)
        self.X = np.array(self.X)
        self.X = self.X.reshape(n_imgs,self.n,self.n,1)
        self.X = self.X.astype('float32') / 255;
        self.labels = np.array(self.labels);
        self.labels = np_utils.to_categorical(self.labels)
        #self.labels = np.array(self.labels);
        #self.labels = self.labels.reshape(n_imgs,8)
    
    
    def extractPatch(self, img):
        patch = np.zeros((self.n,self.n))
        count = 0
        while(np.sum(patch == 0) > self.tolerance*patch.size):
            x = min(randint(1, self.W), self.W - self.n)
            y = min(randint(1, self.H), self.H - self.n)
            patch = img[x:x+self.n, y:y+self.n]
            count = count + 1
            if count == 100:
                break
        return x,y, patch
    ## Derives random patches from an image
    #
    # @param patient_directoy the directory where the specific patient data is located (e.g. Patient_001_Data)
    # @param img the image to derive patches from
    # @param file the patient image number (e.g. img_1)
    def deriveRandomPatch(self, patient_directory,img, file):
        self.lock.acquire()
        x,y,patch = self.extractPatch(img)
        self.X.append(patch)
        self.derivePatchFromSegments(patient_directory, x,y, file)
        self.lock.release()

    ## Derives random patches from an image - Updated for February "pivot"
    #
    # @param patient_directoy the directory where the specific patient data is located (e.g. Patient_001_Data)
    # @param img the image to derive patches from
    # @param file the patient image number (e.g. img_1)
    def deriveRegionsOfInterest(self, patient_directory, data_directory,img, file):
        segment = self.combineSegments(patient_directory, file);
        region = img*segment;
        label_dir = data_directory + '/Ground_Truth/' 
        label_dir = label_dir.encode() + patient_directory[len(patient_directory)-20:len(patient_directory)]
        if not os.path.exists(label_dir):
            return
        label_img = self.getImage(label_dir + file)
        if(np.sum(label_img == 0) <= 0.9*label_img.size):
            for _ in range(0,self.numPatches):
            #patch = np.zeros((self.n, self.n))
            #while np.sum(patch == 0) > self.tolerance*patch.size:
                x,y,patch = self.extractPatch(label_img)
            ## if the entire image is background, we could be stuck in an infinite loop
            ## this mitigates that problem (presumably)
            ## Note to self: refactor this at some point
                #if(np.sum(label_img == 0) > 0.9*label_img.size):
                    #break;
                self.labels.append(int(patch[floor(self.n/2),floor(self.n/2)]/255))
                self.X.append(region[x:x+self.n, y:y+self.n])
        #self.X.append(patch)
        
        #print(label_dir + patient_directory[len(patient_directory)-20:len(patient_directory)])
            
            

    ## Derives and labels the patches in the segment imiage
    #
    # @param patient_dir the specific patient directory (e.g. Patient_001_Data)
    # @param x the starting point for columns (x-direction) for the patch
    # @param y the starting point for rows (y-direction) for the patch
    # @param img_num image number (e.g. img_1)
    # @return a boolean flag which states if a label for the segment was suceesfully found
    def combineSegments(self, patient_dir, img_num):
        segment_directory = os.fsencode(patient_dir) + b'/Segmented_Img_Data'
        seg = np.zeros((self.W, self.H))
        for file in os.listdir(segment_directory+b'/'+img_num[0:len(img_num)-4]):
            seg_num = int(file[4:5].decode("utf-8"))
            if(seg_num > 5):
                seg = np.add(seg, self.getImage(segment_directory+b'/'+img_num[0:len(img_num)-4]+b'/'+file))
        return seg  
                   
    ## Derives an individual patch from an image
    #
    # @param img the image to derive patches from
    # @param stepSize the amount the sliding window shifts per iteration
    # @param file the patient image number (e.g. img_1)
    # @return patches a list of all patches in the image
    def derivePatches(self, img, stepSize):
    # slide a window across the image
        patches = []
        for y in range(0, img.shape[0], stepSize):
            for x in range(0, img.shape[1], stepSize):
            # yield the current window
                window = img[y:y + self.n, x:x + self.n]
                if window.shape[0] != self.n or window.shape[1] != self.n:
                    continue
                else:
                    patches.append((x, y,img[y:y + self.n, x:x + self.n]))
        return patches
    
    ## Derives and labels the patches in the segment imiage
    #
    # @param patient_dir the specific patient directory (e.g. Patient_001_Data)
    # @param x the starting point for columns (x-direction) for the patch
    # @param y the starting point for rows (y-direction) for the patch
    # @param img_num image number (e.g. img_1)
    # @return a boolean flag which states if a label for the segment was suceesfully found
    def derivePatchFromSegments(self, patient_dir, x ,y, img_num):
        segment_directory = os.fsencode(patient_dir) + b'/Segmented_Img_Data'
        for file in os.listdir(segment_directory+b'/'+img_num[0:len(img_num)-4]):
            seg_img = self.getImage(segment_directory+b'/'+img_num[0:len(img_num)-4]+b'/'+file)
            seg_num = int(file[4:5].decode("utf-8"))
            seg_patch = seg_img[x:x+self.n, y:y+self.n]
            if(seg_patch[floor(self.n/2),floor(self.n/2)] == 255):
                label = seg_num - 1
                self.labels.append(label)
                return True
        return False
                    

    ## Acquires the data from the DataHandler after all data has been loaded and processed
    #
    # @param img_num image number (e.g. img_1)
    # @return the data and the labels for the loaded and processed data
    def getData(self):
        self.preprocessForNetwork()
        return self.X, self.labels
    
    ## Clears the data and labels
    #
    def clearVectors(self):
        self.X =[]
        self.labels = []

        
        
