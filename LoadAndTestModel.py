'''
Created on Nov 28, 2017

@author: daniel
'''


from keras.models import load_model
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from DataHandler import DataHandler
from math import floor




data_dir = '/media/daniel/ExtraDrive1/Patient_Data_Images';
model = load_model('/home/daniel/eclipse-workspace/MRIMath/Models/2018-01-25_20_55/model.h5');
dataHandler = DataHandler()
segments = []
for m in range(0, 8):
    segments.append(np.zeros((240,240)))
for i in range(152, 153):
    patient_directory = os.fsencode(dataHandler.getDirectoryFromIndex(i, data_dir))
    orig_data = patient_directory + b'/Original_Img_Data'
    seg_data = patient_directory + b'/Segmented_Img_Data'
    for file in os.listdir(orig_data):
        print(orig_data+b'/'+file)
        img = dataHandler.getImage(orig_data+b'/'+file)
        patches = dataHandler.derivePatches(img, 1)
        for j in range(0,len(patches)):
            x, y, patch = patches[j]
            patch = patch.reshape(1,25, 25, 1)
            patch = patch/255
            pred = model.predict(patch)
            label = np.argmax(pred)
            if(y+floor(dataHandler.n/2) < img.shape[0] and x+floor(dataHandler.n/2) < img.shape[1] and label < 8):
                segments[label][y + floor(dataHandler.n/2),x+floor(dataHandler.n/2)] = 255
            else:
                print(str(label))
        fig = plt.figure()
        plt.gray();   
        a=fig.add_subplot(1,9,1)
        plt.imshow(img)
        for j in range(2,10):         
            a=fig.add_subplot(1,9,j)
            plt.imshow(segments[j-2])
        a=fig.add_subplot(2,9,1)
        plt.imshow(img)
        ind = 2;
        for k in range(1,9):
            a=fig.add_subplot(2,9,ind)
            plt.imshow(dataHandler.getImage(seg_data + b'/' + file[0:len(file)-4] + b'/seg_' + str(k).encode('UTF-8') + b'.png'))
            ind = ind+1
        plt.show();

            

            
            
            
#X_test = []





