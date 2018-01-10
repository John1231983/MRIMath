from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, GlobalAveragePooling2D
from keras.models import Model
import numpy as np
from keras.callbacks import TensorBoard
import os
import cv2


dims = 240;

def get_im(path):
    path=path.decode()
    img = cv2.imread(path,0)
    #resized = cv2.resize(img, (128, 96))
    return img




S = 2;
F = 3;
input_img = Input(shape=(dims, dims,1))  


x = Conv2D(35, (F, F), activation='relu', padding='same')(input_img)
#x = MaxPooling2D((S, S), padding='same')(x)
x = Conv2D(25, (F, F), activation='relu', padding='same')(x)
#x = MaxPooling2D((S, S), padding='same')(x)
x = Conv2D(10, (F, F), activation='relu', padding='same')(x)
encoded = MaxPooling2D((S, S), padding='same')(x)
#encoded = Flatten()(encoded);
#encoded = GlobalAveragePooling2D()(encoded)
print(encoded.shape)
decoded = Dense(dims*dims, activation='relu')(encoded)


training, segments = load_data('/media/daniel/ExtraDrive1/Patient_Data_Images', 1, 60)
testing, segments2 = load_data('/media/daniel/ExtraDrive1/Patient_Data_Images',61,81)

segmentation_bank = [[] for _ in range(8)]
for i in range(1,2):
    print('Training network: ' + str(i))
    segmentation_bank[i] = Model(input_img, decoded)
    segmentation_bank[i].compile(optimizer='adam', loss='mean_squared_error')
    n_imgs = len(training)
    training=np.array(training)
    training = training.astype('float32') / 255;
    training =training.reshape(n_imgs,dims,dims,1)
    segments[i] = np.array(segments[i]);
    segments[i] =segments[i].reshape(n_imgs,dims*dims)
    n_imgs2 = len(testing)
    testing = np.array(testing)
    testing= testing.astype('float32') / 255;
    testing =testing.reshape(n_imgs2,dims,dims,1)
    segments2[i] = np.array(segments2[i]);
    segments2[i] =segments2[i].reshape(n_imgs2,dims*dims)
    segmentation_bank[i].fit(training, segments[i],
                epochs=40,
                batch_size=50,
                shuffle=True,
                validation_data=(testing, segments2[i]),
                callbacks=[TensorBoard(log_dir='/tmp/segment_data')])
    # serialize model to JSON
    segmentation_bank[i].save('/media/daniel/ExtraDrive1/model_' + str(i) + '_seg2.h5')








