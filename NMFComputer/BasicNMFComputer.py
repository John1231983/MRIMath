'''
Created on Jul 16, 2018

@author: daniel
'''
from NMFComputer.NMFComputer import NMFComputer

import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from numpy import linalg


class BasicNMFComputer(NMFComputer):

    num_iterations = 0
    
    def __init__(self, block_dim = 20, num_hist_bins = 256, num_components = 8, num_iterations = 50):
        super().__init__(block_dim, num_hist_bins, num_components)
        self.num_iterations = num_iterations
        
        
    def cost(self, V, W, H):
        WH = np.matmul(W, H)
        V_WH = V-WH
        return linalg.norm(V_WH, 'fro')
    
    def update(self, V, W, H, V_over_WH):
        epsilon = 1e-9
        H *= (np.dot(V_over_WH.T, W) / W.sum(axis=0)).T

        WH = W.dot(H) + epsilon
        V_over_WH = V / WH
        
        W *= np.dot(V_over_WH, H.T) / H.sum(axis=1)

        WH = W.dot(H) + epsilon
        V_over_WH = V / WH
        
        return W, H, WH, V_over_WH
    
    def computeNMF(self, V):
        #plot =[]
        avg_V = V.mean()
        n, m = V.shape
        W = np.random.random(n * self.num_components).reshape(n, self.num_components) * avg_V
        H = np.random.random(self.num_components * m).reshape(self.num_components, m) * avg_V
        WH = W.dot(H)

        V_over_WH = V / WH
    
        for _ in range(0, self.num_iterations):
            W, H, WH, V_over_WH = self.update(V, W, H, V_over_WH)
            #plot.append(self.cost(V, W, H))
        #self.plotCost(plot)
        return W, H
    
    def plotCost(self, data):
        plt.plot(data) 
        plt.xlabel('Iteration')
        plt.ylabel('Frobenius Norm')
        plt.title('NMF Loss')
        plt.show()

            


           
                

        
        
        
        
