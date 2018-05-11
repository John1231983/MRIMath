'''
Created on May 7, 2018

@author: daniel
'''
import os
import numpy as np
import nibabel as nib
import matplotlib .pyplot as plt

LGG_data_dir = "/media/daniel/Backup Data/BRATS_2018/LGG"
for subdir, dirs, files in os.walk(LGG_data_dir):
    for s in os.listdir(subdir):
        print(s)
data_path = "/media/daniel/Backup Data/BRATS_2018/LGG/Brats18_2013_1_1/Brats18_2013_1_1_seg.nii.gz"
img = nib.load(data_path)
print(img.shape)
img = img.get_data()
print(img.shape)
img = img[:,:,48]
print(img)
#super_threshold_indices = img < 2
#img[super_threshold_indices] = 0
#img.reshape(img.shape[0], img.shape[1])
plt.imshow(img)
plt.show() 