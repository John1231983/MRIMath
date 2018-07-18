'''
Created on Jul 16, 2018

@author: daniel
'''
from NMFComputer.NMFComputer import NMFComputer

import numpy as np
from functools import partial
import matplotlib.pyplot as plt


class ProbabilisticNMFComputer(NMFComputer):
    
    def __init__(self, block_size = 400, num_hist_bins = 256, num_components = 5):
        super().__init__(block_size, num_hist_bins, num_components)
    
    def computeNMF(self, V):
        foo = []
        sigma = np.var(V)
        W = np.random.rand(V.shape[0], self.num_components).astype('float128')
        H = np.random.rand(self.num_components, V.shape[1]).astype('float128')
        for _ in range(0, 10):
                
            lamda_H = sigma/np.var(H)
            lamda_W  = sigma/np.var(W)

                
            H_norm = np.matmul(H, H.T)
            W_norm = np.matmul(W.T, W)

                
            W_mag = np.matmul(W.T, V)
            H_mag = np.matmul(V, H.T)

                
            prod_1 = np.matmul(W_norm, H)
            prod_2 = np.matmul(W, H_norm)

            H = np.nan_to_num(np.multiply(H,np.divide(W_mag,(prod_1 + lamda_H*H)))).astype('float128')
            W = np.nan_to_num(np.multiply(W, np.divide(H_mag,(prod_2 + lamda_W*W)))).astype('float128')
        """ 
        plt.figure()
        plt.plot(foo)
        plt.xlabel("k") 
        plt.ylabel("Loss") 
        plt.show()
        """
        return W, H

            


           
                

        
        
        
        
