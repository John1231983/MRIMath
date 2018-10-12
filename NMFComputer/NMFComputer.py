'''
Created on Jul 8, 2018

@author: daniel
'''
import numpy as np
import matplotlib.pyplot as plt


class NMFComputer():
    block_dim = 0
    num_hist_bins = 0
    num_components = 0
    
    def __init__(self, block_dim = 20, num_hist_bins = 256, num_components = 8):
        self.setBlockDim(block_dim)
        self.setNumHistBins(num_hist_bins)
        self.setNumComponents(num_components)

    
    def setBlockDim(self, block_dim):
        if block_dim > 0:
            self.block_dim = block_dim
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
            
    
    def computeHistograms(self, matrix):
        rows = np.vsplit(matrix, matrix.shape[0]/self.block_dim)
        col_split = [np.hsplit(r,matrix.shape[0]/self.block_dim) for r in rows]
        blocks = [item for sublist in col_split for item in sublist]
        hist_image = [np.histogram(block,bins=self.num_hist_bins)[0] for block in blocks] 
        return np.array(hist_image).transpose()
    
    """
    def computeHistograms(self, matrix):
        hist_image = []
        cols = np.hsplit(matrix, self.block_dim)
        print(len(cols))
        for c in cols:
            blocks = np.vsplit(c, self.block_dim)
            print(len(blocks))

            for b in blocks:
                hist, _ = np.histogram(block,bins=self.num_hist_bins)
                hist_image.append(hist)
        return np.array(hist_image).transpose()
    """
    def computeNMF(self, V):
        pass
    
    def showHistogram(self, block):
        fig = plt.figure()
        plt.gray();
        fig.add_subplot(1,2,1)
        plt.imshow(block)
        plt.axis('off')
        plt.title('Original')
        fig.add_subplot(1,2,2)
        hist, bin_edges = np.histogram(block,bins=list(range(self.num_hist_bins)))
        plt.bar(bin_edges[:-1], hist, width = 1)
        plt.title('Histogram')
        #plt.xlim(min(bin_edges), max(bin_edges))
        plt.show()  
        
    def run(self, image):
        V = self.computeHistograms(image)
        W, H = self.computeNMF(V)
        return W, H
