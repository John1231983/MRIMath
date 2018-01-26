'''
Created on Jan 9, 2018

@author: daniel
'''

from TimerModule import TimerModule
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LeakyReLU, PReLU
from keras.models import Sequential
from keras.optimizers import SGD
import os
from EmailHandler import EmailHandler
from HardwareHandler import HardwareHandler
from datetime import datetime
from keras.utils.training_utils import multi_gpu_model
#import tensorflow as tf
from DataHandler import DataHandler
import keras.backend as K
import tensorflow as tf


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

now = datetime.now()
date_string = now.strftime('%Y-%m-%d_%H_%M')
dataHandler = DataHandler()
emailHandler = EmailHandler()
hardwareHandler = HardwareHandler()
timer = TimerModule()

# model = ConvolutionalEncoder([120,60,30,15,15,30,60,120])
# input_img, output = model.getModel()
with tf.device('/cpu:0'):
    input_img = shape=(dataHandler.n, dataHandler.n, 1)
    model = Sequential()
    model.add(Conv2D(175, (3, 3), input_shape=input_img, padding='same'))
    model.add(PReLU())
    model.add(Conv2D(150, (3, 3), padding='same'))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(520))
    model.add(PReLU())
    model.add(Dense(300))
    model.add(PReLU())
# model.add(Conv2D(64, (3, 3), input_shape=input_img, padding='same'))
# model.add(LeakyReLU(0.33))
# model.add(Conv2D(64, (3, 3), input_shape=input_img, padding='same'))
# model.add(LeakyReLU(0.33))
# model.add(Conv2D(64, (3, 3), input_shape=input_img, padding='same'))
# model.add(LeakyReLU(0.33))
# model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
# model.add(Conv2D(128, (3, 3), input_shape=input_img, padding='same'))
# model.add(LeakyReLU(0.33))
# model.add(Conv2D(128, (3, 3), input_shape=input_img, padding='same'))
# model.add(LeakyReLU(0.33))
# model.add(Conv2D(128, (3, 3), input_shape=input_img, padding='same'))
# model.add(LeakyReLU(0.33))
# model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
# model.add(Flatten())
# model.add(Dense(256))
# model.add(LeakyReLU(0.33))
# model.add(Dense(256))
# model.add(LeakyReLU(0.33))
#model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

data_dir = '/coe_data/MRIMath/MS_Research/Patient_Data_Images'
#data_dir = '/media/daniel/ExtraDrive1/Patient_Data_Images'
dataHandler.loadDataParallel(data_dir, 1, 151)
training, training_labels = dataHandler.getData()
dataHandler.clearVectors()
dataHandler.loadDataParallel(data_dir, 151, 192)
testing, testing_labels = dataHandler.getData()

model_directory = "/coe_data/MRIMath/MS_Research/MRIMath/Models/" + date_string
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
    
G = hardwareHandler.getAvailableGPUs()
num_epochs = 40
batchSize = 64
lrate = 0.1
momentum = 0.9
#decay = lrate/num_epochs   
sgd = SGD(lr=lrate, momentum=momentum, nesterov=True)
model_info_filename = 'model_info.txt'
model_info_file = open(model_directory + '/' + model_info_filename, "w") 
log_info_filename = 'model_loss_log.csv'
log_info = open(model_directory + '/' + log_info_filename, "w")
print('Training network!')

#checkpoint = ModelCheckpoint(model_directory + '/checkpoint_weights.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger(model_directory + '/' + log_info_filename, append=True, separator=',')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
print('Using ' + str(G) + ' GPUs to train the network!')
if G > 1:
    parallel_model = multi_gpu_model(model, G)

    parallel_model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics = ['accuracy', precision])
    timer.startTimer()       
    parallel_model.fit(training, training_labels,
        epochs=num_epochs,
        batch_size=batchSize * G,
        shuffle=True,
        validation_data=(testing, testing_labels),
        callbacks=[csv_logger, reduce_lr])
    timer.stopTimer()
    model.set_weights(parallel_model.get_weights())

else:
    timer.startTimer()       
    model.fit(training, training_labels,
        epochs=num_epochs,
        batch_size=batchSize * G,
        shuffle=True,
        validation_data=(testing, testing_labels),
        callbacks=[csv_logger, reduce_lr])
    timer.stopTimer()
        
print('Saving model to disk!')
model.save(model_directory + '/model.h5')
emailHandler.connectToServer()
message = "Finished training network at " + str(datetime.now()) + '\n\n'
message += 'The network was trained on ' + str(training.shape[0]) + ' images \n\n'
message += 'The network was validated on ' + str(testing.shape[0]) + ' images \n\n'
message += "The network was trained for " + str(num_epochs) + " epochs with a batch size of " + str(batchSize) + '\n\n'
message += "The network was saved to " + model_directory + '\n\n'
message += "Total training time: " + str(timer.getElapsedTime())
model.summary(print_fn=lambda x: model_info_file.write(x + '\n'))
emailHandler.prepareMessage(now.strftime('%Y-%m-%d') + " MRIMath Update: Network Training Finished!", message);
model_info_file.close()
emailHandler.attachFile(model_info_file, model_info_filename)
emailHandler.attachFile(log_info, log_info_filename)
emailHandler.sendMessage(["Danny", "Dr.Bouaynaya","Dr.Hassan","Dr.Rasool"])
emailHandler.finish()

