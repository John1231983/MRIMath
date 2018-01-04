'''
Created on Jan 1, 2018

@author: daniel
'''

import os
import cv2

class DataHandler:

    def get_im(self, path):
        path=path.decode()
        img = cv2.imread(path,0)
        return img

    def load_data(self, training_directory, start, finish):
        X_train = []
        segments = [[] for _ in range(8)]
        print('Reading images')
        for j in range(start,finish):
            print('Reading Patient ' + str(j))
            if j < 10:
                directory = os.fsencode(training_directory + '/Patient_(00' + str(j)  + ')_data/')
            elif j < 100:
                directory = os.fsencode(training_directory + '/Patient_(0' + str(j)  + ')_data/')
            else:
                directory = os.fsencode(training_directory + '/Patient_(' + str(j)  + ')_data/')
            for file in os.listdir(directory + b'/Original_Img_Data'):
                img = self.get_im(directory+b'/Original_Img_Data/'+file)
                height, width, channels = img.shape
                if height != 240 or width != 240:
                    break
                X_train.append(img)
            segment_directory = os.fsencode(directory + b'Segmented_Img_Data')
            for dir in os.listdir(segment_directory):
                for file in os.listdir(segment_directory+b'/'+dir):
                    ind = file[4:5]
                    segments[int(ind.decode())-1].append(self.get_im(segment_directory+b'/'+dir+b'/'+file));

        return X_train, segments