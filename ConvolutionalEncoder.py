
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import os
from EmailHandler import EmailHandler
from datetime import datetime
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
from DataHandler import DataHandler
from TimerModule import TimerModule
from keras.callbacks import CSVLogger

import multiprocessing
from joblib import Parallel, delayed

now = datetime.now()
date_string = now.strftime('%Y_%m_%d')


dataHandler = DataHandler()
emailHandler = EmailHandler()
timer = TimerModule()
num_cores = multiprocessing.cpu_count()

F = 3
S = 2 
input_img = Input(shape=(dataHandler.W, dataHandler.H, 1))  

x = Conv2D(120, (F, F), activation='relu', padding='same')(input_img)
x = Conv2D(80, (F, F), activation='relu', padding='same')(x)
x = Conv2D(60, (F, F), activation='relu', padding='same')(x)
x = Conv2D(40, (F, F), activation='relu', padding='same')(x)
encoded = MaxPooling2D((S, S), padding='same')(x)
x = UpSampling2D((S, S))(encoded)
x = Conv2D(40, (F, F), activation='relu', padding='same')(x)
x = Conv2D(60, (F, F), activation='relu', padding='same')(x)
x = Conv2D(80, (F, F), activation='relu', padding='same')(x)
x = Conv2D(120, (F, F), activation='relu', padding='same')(x)
decoded = Conv2D(1, (F, F), activation='relu', padding='same')(x)

emailHandler = EmailHandler()
timer = TimerModule()

#training, segments = Parallel(n_jobs=num_cores)(delayed(dataHandler.loadData)('/coe_data/MRIMath/MS_Research/Patient_Data_Images',i,151) for i in range(151))
training, segments = dataHandler.loadDataParallel('/coe_data/MRIMath/MS_Research/Patient_Data_Images', 1, 151)
testing, segments2 = dataHandler.loadData('/coe_data/MRIMath/MS_Research/Patient_Data_Images',151,192)

model_directory = "/coe_data/MRIMath/MS_Research/MRIMath/Models/" + date_string
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
    
G = 4
num_epochs = 50
batchSize = 32
segmentation_bank = [[] for _ in range(8)]
for i in range(0,8):
    
    specific_model_directory = model_directory + '/' + 'Model ' + str(i)
    if not os.path.exists(specific_model_directory):
        os.makedirs(specific_model_directory)
    
    model_info_filename = 'model_'+str(i) +"_"+ "info.txt"
    model_info_file = open(specific_model_directory + '/' + model_info_filename,"w") 
    log_info_filename = 'model_' + str(i) + '_loss_log.csv'
    log_info = open(specific_model_directory + '/' + log_info_filename, "w")
    
    print('Training network: ' + str(i))
    csv_logger = CSVLogger(specific_model_directory + '/' + log_info_filename, append=True, separator=';')
    with tf.device('/cpu:0'):
        segmentation_bank[i] = Model(input_img, decoded)
    parallel_segmentation_bank = multi_gpu_model(segmentation_bank[i], G)
    parallel_segmentation_bank.compile(optimizer='nadam', loss='mean_squared_error')
    train_segment = segments[:,:,:,i:i+1]
    test_segment = segments2[:,:,:,i:i+1]
    timer.startTimer()
    parallel_segmentation_bank.fit(training, train_segment,
            epochs=num_epochs,
            batch_size=batchSize*G,
            shuffle=True,
            validation_data=(testing, test_segment),
            callbacks=[csv_logger])
    timer.stopTimer()
    
    segmentation_bank[i].set_weights(parallel_segmentation_bank.get_weights())
    print('Saving model ' + str(i) + ' to disk!')
    segmentation_bank[i].save(specific_model_directory + '/model_' + str(i) +'.h5')
    
    emailHandler.connectToServer()
    message = "Finished training network " + str(i) + " at " + str(datetime.now()) + '\n'
    message += "The network was trained for " + str(num_epochs) + " epochs with a batch size of " + str(batchSize) + '\n'
    message += "The model was saved to " + specific_model_directory + '\n'
    
    segmentation_bank[i].summary(print_fn=lambda x: model_info_file.write(x + '\n'))
    message += "\n Total training time: " + str(timer.getElapsedTime())
    emailHandler.prepareMessage(date_string + " MRIMath Update: Network Training Finished!", message);
    model_info_file.close()
    emailHandler.attachFile(model_info_file, model_info_filename)
    emailHandler.attachFile(log_info, log_info_filename)
    emailHandler.sendMessage("Danny")
    emailHandler.finish()







