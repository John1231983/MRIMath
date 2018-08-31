'''
Created on Jul 16, 2018

@author: daniel
'''
from NMFComputer.NMFComputer import NMFComputer

import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from numpy import linalg
np.random.seed(1)

class BasicNMFComputer(NMFComputer):

    num_iterations = 0
    
    def __init__(self, block_dim = 20, num_hist_bins = 256, num_components = 8, num_iterations = 50):
        super().__init__(block_dim, num_hist_bins, num_components)
        self.num_iterations = num_iterations
        
        
    def cost(self, V, W, H):
        WH = np.matmul(W, H)
        V_WH = V-WH
        return linalg.norm(V_WH, 'fro')
    
    def update(self, V, W, H):
        H *= np.nan_to_num((np.dot(W.T, V))/ np.dot(np.dot(W.T, W), H))
        W *= np.nan_to_num((np.dot(V, H.T))/ np.dot(np.dot(W, H), H.T))
        return W, H
    
    def computeNMF(self, V):
        #cost =[]
        avg_V = V.mean()
        n, m = V.shape
        W = np.random.random(n * self.num_components).reshape(n, self.num_components) * avg_V
        H = np.random.random(self.num_components * m).reshape(self.num_components, m) * avg_V

    
        for _ in range(0, self.num_iterations):
            W, H  = self.update(V, W, H)
            #cost.append(self.cost(V, W, H))
        #self.plotCost(cost)
        #self.plotHistograms(W, H)
        return W, H
    
    def plotCost(self, data):
        plt.plot(data) 
        plt.xlabel('Iteration')
        plt.ylabel('Frobenius Norm')
        plt.title('NMF Loss')
        plt.show()
        
    def plotHistograms(self, W, H, N = 2):
        fig = plt.figure()
        fig.add_subplot(1,N+1,1)
        W_cols = np.hsplit(W, W.shape[1])
        H_cols = np.hsplit(H, H.shape[1])
        for i in range(self.num_components):
            plt.bar(list(range(self.num_hist_bins)), W_cols[i].T.tolist()[0], label='Region ' + str(i))
        plt.xlabel('Grayscale Value')
        plt.title('Grayscale Regional Distribution')
        plt.legend()

        for i in range(2,N+2):
            fig.add_subplot(1,N+1,i)
            plt.bar(list(range(self.num_components)), H_cols[i].T.tolist()[0])
        plt.ylabel('Regions')
        plt.show()
        

            


           
                

        
        
        
        
