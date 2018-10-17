'''
Created on Aug 29, 2018

@author: daniel
'''
'''
Created on Jul 10, 2018

@author: daniel
'''

#from multiprocessing import Process, Manager
#from keras.utils import np_utils
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from DataHandlers.UNetDataHandler import UNetDataHandler
from datetime import datetime
import matplotlib.pyplot as plt
from createUNet import createUNet
from keras.optimizers import SGD
from keras.callbacks import CSVLogger
from CustomLosses import combinedDiceAndChamfer
from CustomLosses import dice_coef



def main():
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d_%H_%M')
    
    num_training_patients = 10
    num_validation_patients = 1
    
    dataHandler = UNetDataHandler("Data/BRATS_2018/HGG", num_patients = num_training_patients, modes = ["flair"])
    dataHandler.setMode("training")
    dataHandler.loadData()
    dataHandler.preprocessForNetwork()
    x_train = dataHandler.X
    x_seg_train = dataHandler.labels
    dataHandler.clear()
    
    dataHandler.setDataDirectory("Data/BRATS_2018/HGG_Validation")
    dataHandler.setNumPatients(num_validation_patients)
    dataHandler.setMode("validation")

    dataHandler.loadData()
    dataHandler.preprocessForNetwork()
    x_val = dataHandler.X
    x_seg_val = dataHandler.labels
    dataHandler.clear()
    
    dataHandler.setDataDirectory("Data/BRATS_2018/HGG_Testing")
    dataHandler.setNumPatients(1)
    dataHandler.loadData()
    dataHandler.preprocessForNetwork()
    x_test = dataHandler.X
    x_seg_test = dataHandler.labels
    dataHandler.clear()
    

    input_shape = (dataHandler.W,dataHandler.H, len(dataHandler.modes))
    
    unet = createUNet(input_shape =input_shape)
    num_epochs = 10
    lrate = 0.1
    momentum = 0.9
    decay = lrate/num_epochs   
    sgd = SGD(lr=lrate, momentum=momentum, decay=decay, nesterov=True)
    unet.compile(optimizer=sgd, loss=combinedDiceAndChamfer, metrics=[dice_coef])

    model_directory = "/home/daniel/eclipse-workspace/MRIMath/Models/unet_" + date_string
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    log_info_filename = 'model_loss_log.csv'
    csv_logger = CSVLogger(model_directory + '/' + log_info_filename, append=True, separator=',')
    
    unet.fit(x_train, x_seg_train,
                epochs=10,
                batch_size=20,
                shuffle=True,
                validation_data=(x_val, x_seg_val),
                callbacks = [csv_logger],
                )
    
    
    model_info_filename = 'model_info.txt'
    model_info_file = open(model_directory + '/' + model_info_filename, "w") 
    model_info_file.write('Number of Patients (training): ' + str(num_training_patients) + '\n')
    model_info_file.write('Number of Patients (validation): ' + str(num_validation_patients) + '\n')

    #model_info_file.write('Block Dimensions: ' + str(dataHandler.nmfComp.block_dim) + '\n')
    #model_info_file.write('Number of Components (k): ' + str(dataHandler.nmfComp.num_components) + '\n')
    model_info_file.write('\n\n')
    unet.summary(print_fn=lambda x: model_info_file.write(x + '\n'))
    model_info_file.close();
    unet.save(model_directory + '/model.h5')
    
    decoded_imgs = unet.predict(x_test)
    
    n = 100
    for i in range(n):
        fig = plt.figure()
        plt.gray();   
        a=fig.add_subplot(1,3,1)
        plt.imshow(x_test[i,:,:,0])
        plt.axis('off')
        plt.title('Original')
        
        a=fig.add_subplot(1,3,2)
        plt.imshow(x_seg_test[i,:,:,0])
        plt.axis('off')
        plt.title('GT Segment')
        
        a=fig.add_subplot(1,3,3)
        plt.imshow(decoded_imgs[i,:,:,0])
        plt.axis('off')
        plt.title('Predicted Segment')

        plt.show()
    
    

if __name__ == "__main__":
   main() 
