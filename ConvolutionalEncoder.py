
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np
import os
import cv2
from EmailHandler import EmailHandler
from datetime import datetime
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf


def get_im(path):
    path=path.decode()
    img = cv2.imread(path,0)
    return img

def load_data(training_directory, start, finish):
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
            img = get_im(directory+b'/Original_Img_Data/'+file)
            X_train.append(img)
        segment_directory = os.fsencode(directory + b'Segmented_Img_Data')
        for dir in os.listdir(segment_directory):
            for file in os.listdir(segment_directory+b'/'+dir):
                ind = file[4:5]
                segments[int(ind.decode())-1].append(get_im(segment_directory+b'/'+dir+b'/'+file));

    return X_train, segments
   
   
F = 3
S = 2 
W, H = 240
input_img = Input(shape=(W, H, 1))  



x = Conv2D(100, (F, F), activation='relu', padding='same')(input_img)
x = Conv2D(65, (F, F), activation='relu', padding='same')(x)
x = Conv2D(35, (F, F), activation='relu', padding='same')(x)
x = Conv2D(15, (F, F), activation='relu', padding='same')(x)
encoded = MaxPooling2D((S, S), padding='same')(x)
x = UpSampling2D((S, S))(encoded)
x = Conv2D(15, (F, F), activation='relu', padding='same')(x)
x = Conv2D(35, (F, F), activation='relu', padding='same')(x)
x = Conv2D(65, (F, F), activation='relu', padding='same')(x)
x = Conv2D(100, (F, F), activation='relu', padding='same')(x)
decoded = Conv2D(1, (F, F), activation='relu', padding='same')(x)

training, segments = load_data('/coe_data/MRIMath/MS_Research/Patient_Data_Images', 1, 86)

testing, segments2 = load_data('/coe_data/MRIMath/MS_Research/Patient_Data_Images',86,107)

emailHandler = EmailHandler()

model_directory = "/coe_data/MRIMath/MS_Research/MRIMath/Models/" + datetime.year + "_" + datetime.month + "_" + datetime.day
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
    
G = 4
num_epochs = 50
segmentation_bank = [[] for _ in range(8)]
for i in range(0,8):
    print('Training network: ' + str(i))
    n_imgs = len(training)
    training = np.array(training)
    training =training.reshape(n_imgs,W,H,1)
    training = training.astype('float32') / 255;
    segments[i] = np.array(segments[i]);
    segments[i] = segments[i].reshape(n_imgs,W,H,1)
    n_imgs2 = len(testing)
    testing = np.array(testing)
    testing =testing.reshape(n_imgs2,W,H,1)
    testing= testing.astype('float32') / 255;
    segments2[i] = np.array(segments2[i]);
    segments2[i] = segments2[i].reshape(n_imgs2,240,240,1)
    with tf.device('/cpu:0'):
        segmentation_bank[i] = Model(input_img, decoded)
    parallel_segmentation_bank = multi_gpu_model(segmentation_bank[i], G)
    parallel_segmentation_bank.compile(optimizer='nadam', loss='mean_squared_error')
    parallel_segmentation_bank.fit(training, segments[i],
            epochs=num_epochs,
            batch_size=32*G,
            shuffle=True,
            validation_data=(testing, segments2[i]))
    segmentation_bank[i].save(model_directory + '/model_' + str(i) +'.h5')
    emailHandler.connectToServer()
    emailHandler.prepareMessage("Network Training Finished!", "Finished training network " + str(i) + " at " + str(datetime.now()));
    emailHandler.sendMessage("Danny")
    emailHandler.finish()







