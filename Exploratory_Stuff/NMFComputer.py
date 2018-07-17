'''
Created on Jul 8, 2018

@author: daniel
'''
import numpy as np

class NMFComputer():
    row_window_size = 0
    col_window_size = 0
    num_hist_bins = 0
    num_components = 0
    
    def __init__(self, row_window_size = 12, col_window_size = 12, num_hist_bins = 256, num_components = 5):
        self.setColWindowSize(col_window_size)
        self.setRowWindowSize(row_window_size)
        self.setNumHistBins(num_hist_bins)
        self.setNumComponents(num_components)
        
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
    
    def setNumComponents(self, num_components):
        if num_components > 0:
            self.num_components = num_components
        else:
            print("Error: please enter a valid number of components!")
            
    def computeHistograms(self, image):
        hist_image = []
        for r in range(0,image.shape[0], self.row_window_size):
            for c in range(0,image.shape[1], self.col_window_size):
                window = image[r:r+self.row_window_size,c:c+self.col_window_size]
                hist, bin_edge = np.histogram(window,bins=self.num_hist_bins)
                hist_image.append(hist)
        return np.array(hist_image).transpose()
    
    def computeNMF(self, V):
        pass
    
    def run(self, image):
        V = self.computeHistograms(image)
        W, H = self.computeNMF(V)
        return W, H
