'''
Created on Nov 28, 2017

@author: daniel
'''


from keras.models import load_model
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


training_directory = '/media/daniel/ExtraDrive1/Patient_Data_Images';
segment_number = 0;
k = 1;
model = load_model('/home/daniel/eclipse-workspace/MRIMath/Models/model_' + str(segment_number) + '.h5' )
#X_test = []



