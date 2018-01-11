from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from DataHandler import DataHandler



class ConvolutionalSegmenter():
    
    F = 3
    S = 2 
    input_img = Input(shape=(DataHandler.W, DataHandler.H, 1))  
    x = input_img
    output = 0

    
    def __init__(self, filters):
        for ii in range(len(filters)):
            self.x = Conv2D(filters[ii], (self.F, self.F), activation='relu', padding='same')(self.x)
            if(ii == len(filters)-1):
                self.x = MaxPooling2D((self.S, self.S), padding='same')(self.x)
                self.x = Flatten()(self.x)
                self.output = Dense(DataHandler.W*DataHandler.H, activation='relu')(self.x)
        
    
    def getModel(self):
        return self.input_img,self.output














