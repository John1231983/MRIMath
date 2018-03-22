'''
Created on Feb 17, 2018

@author: daniel
'''

from TimerModule import TimerModule
from keras.callbacks import CSVLogger,ReduceLROnPlateau
from keras.layers import Conv2D, Activation, MaxPooling2D, Reshape, Dense, Flatten, BatchNormalization, Dropout, LeakyReLU, PReLU,concatenate
from keras.models import Sequential, Model, Input
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
from Mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D




# eh, I threw this in here for the sake of having another performance metric -
# maybe we don't need it, or maybe I'll make a StatHandler class to hold it
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

#load up data! This will take a few minutes, even parallelized...
dataHandler = DataHandler()
mode = 'Flair' # Flair, T1, T2, or T1c
hardwareHandler = HardwareHandler()
emailHandler = EmailHandler()
timer = TimerModule()
now = datetime.now()
date_string = now.strftime('%Y-%m-%d_%H_%M')
# Creating the model on the CPU
with tf.device('/cpu:0'):
    
    input_shape = (dataHandler.n, dataHandler.n, 1)
    n_labels = (dataHandler.n, dataHandler.n, 1)
    inputs = Input(shape=input_shape)
    kernel = 3
    pool_size = 2
    output_mode = "softmax"
    conv_1 = Conv2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Conv2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Conv2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Conv2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Conv2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Conv2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Conv2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Conv2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Conv2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Conv2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Conv2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Conv2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Conv2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build enceder done..")

    # decoder

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = Conv2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Conv2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Conv2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = Conv2D(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Conv2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Conv2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Conv2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Conv2D(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Conv2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Conv2D(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Conv2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Conv2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Conv2D(n_labels, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Reshape((input_shape[0] * input_shape[1], n_labels), input_shape=(input_shape[0], input_shape[1], n_labels))(conv_26)

    outputs = Activation(output_mode)(conv_26)
    print("Build decoder done..")

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")



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
    

num_epochs = 30  
batchSize = 128 
lrate = 0.1e-3
momentum = 0.9
decay_rate = lrate / num_epochs

#decay = lrate/num_epochs   
sgd = SGD(lr=lrate, momentum=momentum, decay=decay_rate, nesterov=True)
model_info_filename = 'model_info.txt'
model_info_file = open(model_directory + '/' + model_info_filename, "w") 
log_info_filename = 'model_loss_log.csv'
log_info = open(model_directory + '/' + log_info_filename, "w")
print('Training network!')

# Declaring the two callbacks I use - still having issues using the model checkpoint one
# Also considering using the Early Termination...
csv_logger = CSVLogger(model_directory + '/' + log_info_filename, append=True, separator=',')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.1e-9)

# Grab the number of available GPUS on the device you're running on - in the case of the HPC, it's 4
# And on your local lap top, probably 1
# Based on the result, your model may be converted to a multiple GPU model
G = hardwareHandler.getAvailableGPUs()
print('Using ' + str(G) + ' GPUs to train the network!')
if G > 1:
    parallel_model = multi_gpu_model(model, G)
    parallel_model.compile(optimizer=sgd, loss='binary_crossentropy',metrics = ['accuracy', precision])
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
