
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
import math
from DataHandler import DataHandler



class ConvolutionalEncoder():
    
    F = 3
    S = 2 
    input_img = Input(shape=(DataHandler.W, DataHandler.H, 1))  
    x = input_img
    output = 0

    
    def __init__(self, filters):
        for ii in range(len(filters)):
            if(ii == math.ceil(len(filters)/2)):
                self.x = MaxPooling2D((self.S, self.S), padding='same')(self.x)
                self.x = UpSampling2D((self.S, self.S))(self.x)
            self.x = Conv2D(filters[ii], (self.F, self.F), activation='relu', padding='same')(self.x)
            if(ii == len(filters)-1):
                self.output = Conv2D(1, (self.F, self.F), activation='relu', padding='same')(self.x)

        
    
    def getModel(self):
        return self.input_img,self.output






