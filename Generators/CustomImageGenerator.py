'''
Created on Oct 27, 2018

@author: daniel
'''
from Generators.CustomGenerator import CustomGenerator
import numpy as np
from random import shuffle
from keras.utils import np_utils
class CustomImageGenerator(CustomGenerator):
    
    def __init__(self):
        pass
    
    def generate(self, x_train, x_seg, batch_size, n_labels, normalize = True):
        
        mu = np.mean(np.array(x_train))
        sigma = np.std(np.array(x_train))   
        x_seg = [label.reshape(label.shape[0] * label.shape[1]) for label in x_seg]
            
        if n_labels == 1:
            for x in x_seg:
                x[x > 0.5] = 1
                x[x < 0.5] = 0
                    
        if normalize:
            x_train = [(x-mu)/sigma for x in x_train]   
            
        
        while True:
            data = list(zip(x_train, x_seg))
            shuffle(data)
            x_train_shuffled, x_seg_shuffled = zip(*data)
            for i in range(0, int(len(x_train)/batch_size)):
                batch_imgs, batch_labels = zip(*[(x_train_shuffled[j], x_seg_shuffled[j]) for j in range(i*batch_size, (i+1)*batch_size)])
                batch_imgs = np.array(batch_imgs)
                batch_labels = np.array(batch_labels)
                
                if n_labels > 1:
                    batch_labels = np_utils.to_categorical(batch_labels)

                
                yield (batch_imgs, batch_labels)