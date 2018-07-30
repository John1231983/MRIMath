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
    
    def __init__(self, block_size = 400, num_hist_bins = 256, num_components = 8):
        super().__init__(block_size, num_hist_bins, num_components)
        
        
    def cost(A, W, H):
        WH = np.matmul(W, H)
        A_WH = A-WH
        return linalg.norm(A_WH, 'fro')
    
    
    def computeNMF(self, V):
        
        sigma = np.var(V)
        
        W = np.abs(np.random.uniform(low=0, high=1, size=(V.shape[0], self.num_components)))
        H = np.abs(np.random.uniform(low=0, high=1, size=(self.num_components, V.shape[1])))
     
        for _ in range(0, 10):
            lamda_H = sigma/np.var(H)
            lamda_W  = sigma/np.var(W)

            H_norm = np.matmul(H, H.T)
            W_norm = np.matmul(W.T, W)

            WV = np.matmul(W.T, V)
            HV = np.matmul(V, H.T)

            WTWH = np.matmul(W_norm, H)
            WHHT = np.matmul(W, H_norm)
            
            H *= np.divide(WV,WTWH+lamda_H*H)
            W *= np.divide(HV,WHHT+lamda_W*W)
            
            print(self.cost(V, W, H))
            
            
            
        

        return W, H

            


           
                

        
        
        
        
