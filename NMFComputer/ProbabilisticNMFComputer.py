'''
Created on Jul 16, 2018

@author: daniel
'''
from NMFComputer.NMFComputer import NMFComputer

import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from numpy import linalg


class ProbabilisticNMFComputer(NMFComputer):

    num_iterations = 0
    
    def __init__(self, block_dim = 20, num_hist_bins = 256, num_components = 8, num_iterations = 100):
        super().__init__(block_dim, num_hist_bins, num_components)
        self.num_iterations = num_iterations
        
        
    def cost(self, V, W, H):
        WH = np.matmul(W, H)
        V_WH = V-WH
        return linalg.norm(V_WH, 'fro')
    
    
    def computeNMF(self, V):
        sigma = np.var(V)
        plot =[]
        
        W = np.abs(np.random.uniform(low=0, size=(V.shape[0], self.num_components)))
        H = np.abs(np.random.uniform(low=0, size=(self.num_components, V.shape[1])))
    
        for _ in range(0, self.num_iterations):
            lamda_H = sigma/np.var(H)
            lamda_W  = sigma/np.var(W)

            H_norm = np.matmul(H, H.T)
            W_norm = np.matmul(W.T, W)

            WV = np.matmul(W.T, V)
            HV = np.matmul(V, H.T)

            WTWH = np.matmul(W_norm, H)
            WHHT = np.matmul(W, H_norm)
            
            H = np.multiply(H,np.divide(WV,WTWH+lamda_H*H))
            W = np.multiply(W,np.divide(HV,WHHT+lamda_W*W))
            plot.append(self.cost(V, W, H))
	    
        return W, H
    
    def plotCost(self, data):
        plt.plot(data) 
        plt.xlabel('Iteration')
        plt.ylabel('NMF Loss')
        plt.show()

            


           
                

        
        
        
        
