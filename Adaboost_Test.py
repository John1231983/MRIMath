'''
Created on Aug 28, 2018

@author: daniel
'''
'''
Created on Aug 27, 2018

@author: daniel
'''
from sklearn.ensemble import AdaBoostClassifier
from NMFComputer.BasicNMFComputer import BasicNMFComputer
from Exploratory_Stuff.BlockDataHandler import BlockDataHandler
import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
def main():
    print('Loading the data! This could take some time...')
    mode = "flair"
    num_training_patients = 5;
    num_validation_patients = 1;
    nmfComp = BasicNMFComputer(block_dim=8, num_components=25)
    dataHandler = BlockDataHandler("Data/BRATS_2018/HGG", nmfComp, num_patients = num_training_patients)
    dataHandler.loadData(mode)
    #dataHandler.preprocessForNetwork()
    x_train = dataHandler.X
    x_train_size = len(x_train)
    x_train = np.array(x_train).reshape(x_train_size,-1)
    labels = dataHandler.labels
    model = AdaBoostClassifier()
    model.fit(x_train, labels)
    
    """
    dataHandler.clear()
    dataHandler.setDataDirectory("Data/BRATS_2018/HGG_Validation")
    dataHandler.setNumPatients(num_validation_patients)
    dataHandler.loadData(mode)
    """
    #dataHandler.preprocessForNetwork()
    #x_val = dataHandler.X
    #val_labels = dataHandler.labels
    
    #x_val = np.array(x_val).reshape(x_val_size,-1)
    
    test_data_dir = "Data/BRATS_2018/HGG_Validation"
    image = None
    seg_image = None
    for subdir in os.listdir(test_data_dir):
        for path in os.listdir(test_data_dir+ "/" + subdir):
            if mode in path:
                image = nib.load(test_data_dir + "/" + subdir + "/" + path).get_data()
                seg_image = nib.load(test_data_dir+ "/" + subdir + "/" + path.replace(mode, "seg")).get_data()
                break
            
    m = nmfComp.block_dim
    inds = [i for i in list(range(155)) if np.count_nonzero(seg_image[:,:,i]) > 0]
    
    
    for k in inds:
        seg_est = np.zeros(shape=(dataHandler.W, dataHandler.H))
        img = image[:,:,k]
        seg_img = seg_image[:,:,k]
        """
        rmin,rmax, cmin, cmax = dataHandler.bbox(image)
        img = img[rmin:rmax, cmin:cmax]
        img = cv2.resize(img, dsize=(dataHandler.W, dataHandler.H), interpolation=cv2.INTER_LINEAR)
        seg_img = seg_img[rmin:rmax, cmin:cmax]
        seg_img = cv2.resize(seg_img, dsize=(dataHandler.W, dataHandler.H), interpolation=cv2.INTER_LINEAR)
        """
        
        #img = dataHandler.preprocess(image[:,:,k])
        _, H = nmfComp.run(img)
        H_cols = np.hsplit(H, H.shape[1])
        est_labels = [model.predict(x.T) for x in H_cols]
        #print(est_labels)
        gt_labels = dataHandler.getLabels(seg_img)
        #print( str(np.linalg.norm(np.array(gt_labels) - np.argmax(np.array(est_labels), axis=2), 'fro')))
        #labels = model.predict(H.T)
        ind = 0
        for i in range(0, dataHandler.W, m):
            for j in range(0, dataHandler.H, m):
                seg_est[i:i+m, j:j+m] = np.full((m, m), est_labels[ind])
                ind = ind+1
        
        fig = plt.figure()
        plt.gray();
        a=fig.add_subplot(1,3,1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Original')

        a=fig.add_subplot(1,3,2)
        plt.imshow(seg_image[:,:,k])
        plt.axis('off')
        plt.title('GT Segment')
        
        a=fig.add_subplot(1,3,3)
        plt.imshow(seg_est)
        plt.axis('off')
        plt.title('Estimate Segment')
        plt.show()
    
    
if __name__ == "__main__":
   main()  