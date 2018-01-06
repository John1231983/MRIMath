
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np
import os
from EmailHandler import EmailHandler
from datetime import datetime
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
from DataHandler import DataHandler
from TimerModule import TimerModule

now = datetime.now()
date_string = now.strftime('%Y_%m_%d')
   
F = 3
S = 2 
W = 240
H = 240
input_img = Input(shape=(W, H, 1))  

x = Conv2D(120, (F, F), activation='relu', padding='same')(input_img)
x = Conv2D(80, (F, F), activation='relu', padding='same')(x)
x = Conv2D(60, (F, F), activation='relu', padding='same')(x)
x = Conv2D(40, (F, F), activation='relu', padding='same')(x)
x = Conv2D(20, (F, F), activation='relu', padding='same')(x)
encoded = MaxPooling2D((S, S), padding='same')(x)
x = UpSampling2D((S, S))(encoded)
x = Conv2D(20, (F, F), activation='relu', padding='same')(x)
x = Conv2D(40, (F, F), activation='relu', padding='same')(x)
x = Conv2D(60, (F, F), activation='relu', padding='same')(x)
x = Conv2D(80, (F, F), activation='relu', padding='same')(x)
x = Conv2D(120, (F, F), activation='relu', padding='same')(x)
decoded = Conv2D(1, (F, F), activation='relu', padding='same')(x)

emailHandler = EmailHandler()
dataHandler = DataHandler()
timer = TimerModule()

training, segments = dataHandler.load_data('/coe_data/MRIMath/MS_Research/Patient_Data_Images', 1, 4)
testing, segments2 = dataHandler.load_data('/coe_data/MRIMath/MS_Research/Patient_Data_Images',151,152)

model_directory = "/coe_data/MRIMath/MS_Research/MRIMath/Models/" + date_string
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
    
G = 4
num_epochs = 1
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
    timer.startTimer()
    parallel_segmentation_bank.fit(training, segments[i],
            epochs=num_epochs,
            batch_size=32*G,
            shuffle=True,
            validation_data=(testing, segments2[i]))
    timer.stopTimer()
    segmentation_bank[i].set_weights(parallel_segmentation_bank.get_weights())
    print('Saving model ' + str(i) + ' to disk!')
    segmentation_bank[i].save(model_directory + '/model_' + str(i) +'.h5')
    emailHandler.connectToServer()
    message = "Finished training network " + str(i) + " at " + str(datetime.now())
    message += "\n Total training time: " + timer.getElapsedTime()
    message += "\n\n " + segmentation_bank[i].summary()
    emailHandler.prepareMessage("Network Training Finished!", message);
    emailHandler.sendMessage("Danny")
    emailHandler.finish()







