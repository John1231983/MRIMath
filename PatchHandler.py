'''
Created on Jan 17, 2018

@author: daniel
'''
from random import randint
import numpy as np
import os
from math import floor



class PatchHandler:
    
    tolerance = None #the percentage of background pixels permitted in a patch
    numPatches = None #the number of patches to extract from an imaage
    n = None #dimensions of each patch (n x n)
    #stepSize = None # the stride, in the case of a sliding window approach
    
    def __init__(self, tolerance = 0.25, numPatches = 5, n = 25):
        self.tolerance = tolerance
        self.numPatches = numPatches
        self.n = n
        

    def derivePatches(self, img, stepSize):
        for (x, y, window) in self.deriveIndividualPatch(img, stepSize):
            if window.shape[0] != self.n or window.shape[1] != self.n:
                continue
            #self.X.append(window)
        
    def deriveRandomPatch(self, data_directory,img):
        x = min(randint(1, 240), 240 - self.n)
        y = min(randint(1, 240), 240 - self.n)
        patch = img[x:x+self.n, y:y+self.n]
        labels = self.derivePatchesFromSegments(data_directory, x,y)
        numBackgroundPixels = np.sum(patch == 0)
        if numBackgroundPixels > self.tolerance*img.row*img.col:
            return self.deriveRandomPatch(img)
        else:
            return patch, labels
                   
            
    def deriveIndividualPatch(self, img, stepSize, n):
    # slide a window across the image
        for y in range(0, img.shape[0], stepSize):
            for x in range(0, img.shape[1], stepSize):
            # yield the current window
                yield (x, y, img[y:y + n, x:x + n])
                
    def derivePatchesFromSegments(self, data_directory, x ,y):
        labels = []
        for ind in range(1,self.numPatches):
            label = []
            for _ in range(0, 8):
                label.append(0)
            segment_directory = os.fsencode(self.getDirectoryFromIndex(ind, data_directory)) + b'/Segmented_Img_Data'
            for dir in os.listdir(segment_directory):
                for file in os.listdir(segment_directory+b'/'+dir):
                    seg_img = self.getImage(segment_directory+b'/'+dir+b'/'+file)
                    seg_num = file[4:5]
                    seg_patch = seg_img[x:x+self.n, y:y+self.n]
                    if(seg_patch[floor(self.n/2),floor(self.n/2)] == 255):
                        label[seg_num] = 1
                    
                    
                labels.append(label)
        return labels