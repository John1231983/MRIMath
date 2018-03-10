'''
Created on Feb 17, 2018

@author: daniel
'''

from TimerModule import TimerModule
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LeakyReLU, PReLU,concatenate
from keras.models import Sequential, Model
from keras.optimizers import SGD
import os
from EmailHandler import EmailHandler
from HardwareHandler import HardwareHandler
from datetime import datetime
from keras.utils.training_utils import multi_gpu_model
from DataHandler import DataHandler
import keras.backend as K
import tensorflow as tf
import numpy as np



# eh, I threw this in here for the sake of having another performance metric -
# maybe we don't need it, or maybe I'll make a StatHandler class to hold it
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# Basic initialization of some of the handlers...
now = datetime.now()
date_string = now.strftime('%Y-%m-%d_%H_%M')
dataHandler = DataHandler()
emailHandler = EmailHandler()
hardwareHandler = HardwareHandler()
timer = TimerModule()
mode = 'Flair' # Flair, T1, T2, or T1c
# Creating the model on the CPU
with tf.device('/cpu:0'):
    input_img = shape=(dataHandler.n, dataHandler.n, 1)
    tower_1 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
    tower_1 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
    tower_2 = Conv2D(64, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
    tower_3 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_3)
    output = concatenate([tower_1, tower_2, tower_3], axis = 3)
    output = Flatten()(output)
    out    = Dense(10, activation='softmax')(output)
    model = Model(inputs = input_img, outputs = out)
    #model = Sequential()
    #model.add(Conv2D(100, (3, 3), input_shape=input_img, padding='same'))
    #model.add(LeakyReLU())
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Conv2D(100, (3, 3), padding='same'))
    #model.add(LeakyReLU())
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Flatten())
    #model.add(Dense(150))
    #model.add(LeakyReLU())
    #model.add(Dense(75))
    #model.add(LeakyReLU())
    #model.add(Dense(1, activation='sigmoid'))


#load up data! This will take a few minutes, even parallelized...
data_dir = '/coe_data/MRIMath/MS_Research/Patient_Data_Images/'+mode
#data_dir = '/media/daniel/ExtraDrive1/Patient_Data_Images'
dataHandler.loadDataParallel(data_dir, 1, 151)
training, training_labels = dataHandler.getData()
dataHandler.clearVectors()
dataHandler.loadDataParallel(data_dir, 151, 192)
testing, testing_labels = dataHandler.getData()

# Creates a directory to save everything (model, loss log, and model info)
# Should always be unique since the date string is based on the current date and the time
model_directory = "/coe_data/MRIMath/MS_Research/MRIMath/Models/" + date_string+'_'+mode
#model_directory = "/media/daniel/ExtraDrive1/Patient_Data_Images/Models/" + date_string + mode
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
    

num_epochs = 35  
batchSize = 64
lrate = 0.1e-5
momentum = 0.9
#decay = lrate/num_epochs   
sgd = SGD(lr=lrate, momentum=momentum, nesterov=True)
model_info_filename = 'model_info.txt'
model_info_file = open(model_directory + '/' + model_info_filename, "w") 
log_info_filename = 'model_loss_log.csv'
log_info = open(model_directory + '/' + log_info_filename, "w")
print('Training network!')

# Declaring the two callbacks I use - still having issues using the model checkpoint one
# Also considering using the Early Termination...
csv_logger = CSVLogger(model_directory + '/' + log_info_filename, append=True, separator=',')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.1e-9)

# Grab the number of available GPUS on the device you're running on - in the case of the HPC, it's 4
# And on your local lap top, probably 1
# Based on the result, your model may be converted to a multiple GPU model
G = hardwareHandler.getAvailableGPUs()
print('Using ' + str(G) + ' GPUs to train the network!')
if G > 1:
    parallel_model = multi_gpu_model(model, G)
    parallel_model.compile(optimizer='adagrad', loss='binary_crossentropy',metrics = ['accuracy', precision])
    timer.startTimer()   # this is for timing/profiling purposes    
    parallel_model.fit(training, training_labels,
        epochs=num_epochs,
        batch_size=batchSize * G,
        shuffle=True,
        validation_data=(testing, testing_labels),
        callbacks=[csv_logger, reduce_lr])
    timer.stopTimer()
    model.set_weights(parallel_model.get_weights())

else:
    model.compile(optimizer=sgd, loss='binary_crossentropy',metrics = ['accuracy', precision])
    timer.startTimer()
    model.fit(training, training_labels,
        epochs=num_epochs,
        batch_size=batchSize * G,
        shuffle=True,
        validation_data=(testing, testing_labels),
        callbacks=[csv_logger, reduce_lr])
    timer.stopTimer()
  
# Wrap Up - Just some useful screen printouts and constructing + sending an email to notify selected individuals
# that training has finished, and saving the relevant files to the directory we created earlier    
print('Saving model to disk!')
model.save(model_directory + '/model.h5')
emailHandler.connectToServer()
message = "Finished training network at " + str(datetime.now()) + '\n\n'
message += 'The network was trained on ' + str(training.shape[0]) +' '+ mode +  ' images \n\n'
message += 'The network was validated on ' + str(testing.shape[0]) +' '+ mode +  ' images \n\n'
message += "The network was trained for " + str(num_epochs) + " epochs with a batch size of " + str(batchSize) + '\n\n'
message += "The network was saved to " + model_directory + '\n\n'
message += "Total training time: " + str(timer.getElapsedTime())
model.summary(print_fn=lambda x: model_info_file.write(x + '\n'))
emailHandler.prepareMessage(now.strftime('%Y-%m-%d') + " MRIMath Update: Network Training Finished!", message);
model_info_file.close()
emailHandler.attachFile(model_info_file, model_info_filename)
emailHandler.attachFile(log_info, log_info_filename)
emailHandler.sendMessage(["Danny"])
emailHandler.finish()

