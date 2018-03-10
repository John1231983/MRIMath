'''
Created on Mar 10, 2018

@author: daniel
'''
from keras.models import load_model
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from DataHandler import DataHandler
from math import floor


data_dir = '/media/daniel/Backup Data/Flair';
model = load_model('/home/daniel/eclipse-workspace/MRIMath/Models/2018-03-08_09_39_Flair/model.h5');
dataHandler = DataHandler()
for i in range(152, 153):
    patient_directory = os.fsencode(dataHandler.getDirectoryFromIndex(i, data_dir))
    orig_data = patient_directory + b'/Original_Img_Data'
    ground_truth = b'/media/daniel/Backup Data/Flair/Ground_Truth/' + os.fsencode(dataHandler.getPatientDirectoryFromIndex(i))
    for file in os.listdir(orig_data):
        print(orig_data+b'/'+file)
        true_segment = dataHandler.getImage(ground_truth+b'/'+file)
        img = dataHandler.getImage(orig_data+b'/'+file)
        patches = dataHandler.derivePatches(img, 1)
        predicted_segment = np.zeros((img.shape[0], img.shape[1]))
        for j in range(0,len(patches)):
            x, y, patch = patches[j]
            patch = patch.reshape(1,dataHandler.n, dataHandler.n, 1)
            patch = patch/255
            pred = model.predict(patch)
            label = np.rint(pred)
            predicted_segment[y + floor(dataHandler.n/2),x+floor(dataHandler.n/2)] = label
        intersection = np.logical_and(predicted_segment, true_segment)
        union = np.logical_or(predicted_segment, true_segment)
        print(str(np.sum(intersection)/np.sum(union)))
        predicted_segment *= 255
        fig = plt.figure()
        plt.gray();   
        a=fig.add_subplot(1,3,1)
        plt.imshow(img)
        a=fig.add_subplot(1,3,2)
        plt.imshow(predicted_segment)
        a=fig.add_subplot(1,3,3)
        plt.imshow(true_segment)
        plt.show();

            

            
            
            
#X_test = []





