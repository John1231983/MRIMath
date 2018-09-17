'''
Created on Jun 18, 2018

@author: daniel
'''
from MRIMathDataset import MRIMathDataset

class FlairDataset(MRIMathDataset):

    
    def __init__(self):
        super().__init__()
        self.mode = "flair"
        self.tumor_type = "whole"
    
    
    def getMask(self, mask):
        mask[mask > 0] = 1
        return mask