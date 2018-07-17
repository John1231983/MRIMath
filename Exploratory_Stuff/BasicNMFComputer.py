'''
Created on Jul 16, 2018

@author: daniel
'''

from Exploratory_Stuff.NMFComputer import NMFComputer
from sklearn.decomposition import NMF


class BasicNMFComputer(NMFComputer):
    nmf_model = None

    def __init__(self, row_window_size = 12, col_window_size = 12, num_hist_bins = 256, num_components = 5):
        super().__init__(row_window_size, col_window_size, num_hist_bins, num_components)
        self.creatNMFModel(num_components)

        
        
    def computeNMF(self, V):
        W = self.nmf_model.fit_transform(V)
        H = self.nmf_model.components_
        print(W.shape)
        print(H.shape)
        return W, H
    
        
    def creatNMFModel(self, num_components):
        self.nmf_model = NMF(n_components=num_components, init='random', random_state=0)
            
