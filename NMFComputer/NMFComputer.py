'''
Created on Jul 8, 2018

@author: daniel
'''
import numpy as np
import math
class NMFComputer():
    block_size = 0
    num_hist_bins = 0
    num_components = 0
    
    def __init__(self, block_size = 400, num_hist_bins = 256, num_components = 5):
        self.setBlockSize(block_size)
        self.setNumHistBins(num_hist_bins)
        self.setNumComponents(num_components)

    
    def setBlockSize(self, block_size):
        if block_size > 0:
            self.block_size = block_size
        else:
            print("Error: please enter a valid column window size!")

    
    def setNumHistBins(self, num_hist_bins):
        if num_hist_bins > 0:
            self.num_hist_bins = num_hist_bins
        else:
            print("Error: please enter a valid histogram bin size!")
    
    def setNumComponents(self, num_components):
        if num_components > 0:
            self.num_components = num_components
        else:
            print("Error: please enter a valid number of components!")
    
    def computeHistogram(self, block):
        hist, _ = np.histogram(block,bins=self.num_hist_bins)
        return hist
    
    def computeHistograms(self, matrix):
        m = math.sqrt(self.block_size)
        cols = np.hsplit(matrix, m)
        blocks = [np.vsplit(c,m) for c in cols]
        hist_image = [self.computeHistogram(block) for block in blocks] 
        print(hist_image)           
        return np.array(hist_image).transpose()
    
    def computeNMF(self, V):
        pass
    
    def run(self, image):
        V = self.computeHistograms(image)
        W, H = self.computeNMF(V)
        return W, H
