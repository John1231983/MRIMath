'''
Created on Oct 23, 2018

@author: daniel
'''
## Custom image data augmentation generator
## for brain MRI images
## Augmentations based on:

import numpy as np
import tensorlayer as tl
from random import random, shuffle

from keras.utils import np_utils

class CustomImageAugmentationGenerator:
    
    alpha = None
    sigma = None
    
    def __init__(self, alpha = 720, sigma = 24):
        self.alpha = alpha
        self.sigma = sigma
        
    
    def augmentData(self, data):
        """ data augumentation """
        foo = data
        
        foo = [np.squeeze(x) for x in foo]

        foo = tl.prepro.elastic_transform_multi(list(foo),
                                alpha=720, sigma=24, is_random=True)
        
        foo = [np.expand_dims(x, axis=-1) for x in foo]
        
        foo = tl.prepro.flip_axis_multi(list(foo),
                        axis=1, is_random=True) # left right
        
        foo = tl.prepro.flip_axis_multi(list(foo),
                axis=0, is_random=True) # up down
        """
        
        foo = tl.prepro.brightness_multi(list(foo), 
                                         0.8, 1, is_random=True)
        """
        foo = tl.prepro.rotation_multi(list(foo), rg=20,
                                is_random=True, fill_mode='constant') # nearest, constant
        
        foo = tl.prepro.shift_multi(list(foo), wrg=0.10,
                                hrg=0.10, is_random=True, fill_mode='constant')
        """
        foo = tl.prepro.shear_multi(foo, 0.05,
                                is_random=True, fill_mode='constant')
        """
        #foo = [np.squeeze(x[1]) for x in foo]
        

        return foo
    
    def generate(self, x_train, x_seg, batch_size, n_labels, normalize = True):
        
        mu = np.mean(np.array(x_train))
        sigma = np.std(np.array(x_train))
        
        while True:
            data = list(zip(x_train, x_seg))
            shuffle(data)
            x_train_shuffled, x_seg_shuffled = zip(*data)
            augmented_data = []
            for i in range(0, int(len(x_train)/batch_size)):
                augmented_data = []
                for j in range(i*batch_size, (i+1)*batch_size):
                    aug_img = self.augmentData([x_train_shuffled[j], x_seg_shuffled[j]])
                    augmented_data.append(aug_img)
                (aug_batch_imgs, aug_batch_labels) = zip(*augmented_data)
                aug_batch_labels = [label.reshape(label.shape[0] * label.shape[1]) for label in aug_batch_labels]
                aug_batch_labels = np.array(aug_batch_labels)
                aug_batch_imgs = np.array(aug_batch_imgs)
                if n_labels > 1:
                    aug_batch_labels = np_utils.to_categorical(aug_batch_labels)
                if normalize:
                    aug_batch_imgs -= mu
                    aug_batch_imgs /= sigma
                    
                yield (np.array(aug_batch_imgs), aug_batch_labels)
            
            

