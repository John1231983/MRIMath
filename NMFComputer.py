'''
Created on Jul 8, 2018

@author: daniel
'''
import numpy as np
from sklearn.decomposition import NMF

class NMFComputer():
    row_window_size = 0
    col_window_size = 0
    num_hist_bins = 0
    nmf_model = None
    
    def __init__(self, row_window_size = 5, col_window_size = 5, num_hist_bins = 256, num_components = 5):
        self.setColWindowSize(col_window_size)
        self.setRowWindowSize(row_window_size)
        self.setNumHistBins(num_hist_bins)
        self.creatNMFModel(num_components)
        
        
    def setRowWindowSize(self, row_window_size):
        if row_window_size > 0:
            self.row_window_size = row_window_size
        else:
            print("Error: please enter a valid row window size!")

    
    def setColWindowSize(self, col_window_size):
        if col_window_size > 0:
            self.col_window_size = col_window_size
        else:
            print("Error: please enter a valid column window size!")

    
    def setNumHistBins(self, num_hist_bins):
        if num_hist_bins > 0:
            self.num_hist_bins = num_hist_bins
        else:
            print("Error: please enter a valid histogram bin size!")
    
    def creatNMFModel(self, num_components):
        if num_components > 0:
            self.nmf_model = NMF(n_components=num_components, init='random', random_state=0)
        else:
            print("Error: please enter a valid number of NMF Components!")
            
    def computeHistograms(self, image):
        hist_image = []
        #hist_image = np.zeros([self.num_hist_bins, image.shape[0]/self.row_window_size * image.shape[1]/self.col_window_size])
        for r in range(0,image.shape[0], self.row_window_size):
            for c in range(0,image.shape[1], self.col_window_size):
                window = image[r:r+self.row_window_size,c:c+self.col_window_size]
                hist, bin_edge = np.histogram(window,bins=self.num_hist_bins)
                hist_image.append(hist)
        return np.array(hist_image).transpose()
    
    def computeNMF(self, V):
        W = self.nmf_model.fit_transform(V)
        H = self.nmf_model.components_
        return W, H
    
    def run(self, image):
        V = self.computeHistograms(image)
        W, H = self.computeNMF(V)
        return W, H
