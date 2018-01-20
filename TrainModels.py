'''
Created on Jan 9, 2018

@author: daniel
'''

from TimerModule import TimerModule
from keras.callbacks import CSVLogger
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Model, Sequential
import os
from EmailHandler import EmailHandler
from HardwareHandler import HardwareHandler
from datetime import datetime
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
from DataHandler import DataHandler

now = datetime.now()
date_string = now.strftime('%Y-%m-%d')
dataHandler = DataHandler()
emailHandler = EmailHandler()
hardwareHandler = HardwareHandler()
timer = TimerModule()
# model = ConvolutionalEncoder([120,60,30,15,15,30,60,120])
# input_img, output = model.getModel()
input_img = shape=(dataHandler.n, dataHandler.n, 1)
model = Sequential()
model.add(Conv2D(75, (3, 3), input_shape=input_img, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

data_dir = '/coe_data/MRIMath/MS_Research/Patient_Data_Images'
#data_dir = '/media/daniel/ExtraDrive1/Patient_Data_Images'
dataHandler.loadDataParallel(data_dir, 1, 107)
training, training_labels = dataHandler.getData()
dataHandler.clearVectors()
dataHandler.loadDataParallel(data_dir, 136, 167)
testing, testing_labels = dataHandler.getData()

model_directory = "/coe_data/MRIMath/MS_Research/MRIMath/Models/" + date_string
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
    
G = hardwareHandler.getAvailableGPUs()
num_epochs = 50
batchSize = 64
    
model_info_filename = 'model_info.txt'
model_info_file = open(model_directory + '/' + model_info_filename, "w") 
log_info_filename = 'model_loss_log.csv'
log_info = open(model_directory + '/' + log_info_filename, "w")
print('Training network!')
csv_logger = CSVLogger(model_directory + '/' + log_info_filename, append=True, separator=';')
print('Using ' + str(G) + ' GPUs to train the network!')
if G > 1:
    #with tf.device('/cpu:0'):
        #segmentation_bank[i] = Model(input_img, output)
    parallel_segmentation_bank = multi_gpu_model(model, G)
    parallel_segmentation_bank.compile(optimizer='sgd', loss='categorical_crossentropy')
    timer.startTimer()
    parallel_segmentation_bank.fit(training, training_labels,
            epochs=num_epochs,
            batch_size=batchSize * G,
            shuffle=True,
            validation_data=(testing, testing_labels),
            callbacks=[csv_logger])
    timer.stopTimer()
        
else:
    #segmentation_bank[i] = Model(input_img, output)
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    timer.startTimer()
    model.fit(training, training_labels,
            epochs=num_epochs,
            batch_size=batchSize * G,
            shuffle=True,
            validation_data=(testing, testing_labels),
            callbacks=[csv_logger])
    timer.stopTimer()
        
model.set_weights(parallel_segmentation_bank.get_weights())
print('Saving model to disk!')
model.save(model_directory + '/model.h5')

emailHandler.connectToServer()
message = "Finished training network at " + str(datetime.now()) + '\n\n'
message += 'The network was trained on ' + str(training.shape[0]) + ' images \n\n'
message += 'The network was validated on ' + str(testing.shape[0]) + ' images \n\n'
message += "The network was trained for " + str(num_epochs) + " epochs with a batch size of " + str(batchSize) + '\n\n'
message += "The model was saved to " + model_directory + '\n\n'
message += "Total training time: " + str(timer.getElapsedTime())
model.summary(print_fn=lambda x: model_info_file.write(x + '\n'))
emailHandler.prepareMessage(now.strftime('%Y-%m-%d') + " MRIMath Update: Network Training Finished!", message);
model_info_file.close()
emailHandler.attachFile(model_info_file, model_info_filename)
emailHandler.attachFile(log_info, log_info_filename)
emailHandler.sendMessage(["Danny"])
emailHandler.finish()

