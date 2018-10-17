'''
Created on Aug 29, 2018

@author: daniel
'''

from DataHandlers.DataHandler import DataHandler
import numpy as np
import cv2
import os
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.transform import resize
import SimpleITK as sitk
class UNetDataHandler(DataHandler):
    modes = None
    mode = None
    def __init__(self,dataDirectory, W = 128, H = 128, num_patients = 3, modes = ["flair", "t1ce", "t1", "t2"]):
        super().__init__(dataDirectory, W, H, num_patients)
        self.modes = modes
    
    def windowIntensity(self, image, min_percent=1, max_percent=99):
    
        sitk_image = sitk.GetImageFromArray(image)
        #sitk_image = sitk.IntensityWindowing(sitk_image, np.percentile(image, 1), np.percentile(image, 99))
        sitk_image = sitk.Cast( sitk_image, sitk.sitkFloat64 )
        corrected_image = sitk.IntensityWindowing(sitk_image, np.percentile(image, min_percent), np.percentile(image, max_percent))
    
        corrected_image = sitk.GetArrayFromImage(corrected_image)
        #corrected_image = exposure.equalize_hist(corrected_image)
        return corrected_image
        
    def performNMFOnSlice(self, image, seg_image, i):
        # image[:,:,i] = self.preprocess(image[:,:,i])        
        W, H = self.nmfComp.run(image[:,:,i])
        return self.processData(image[:,:,i], W,H, seg_image[:,:,i])
    
    def setMode(self, mode):
        self.mode = mode;
    
    def getMode(self):
        return self.mode
    def processImage(self, image):
        rmin,rmax, cmin, cmax = self.bbox(image)
        image = image[rmin:rmax, cmin:cmax]
        resized_image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return resized_image, rmin, rmax, cmin, cmax
        
    def loadData(self):
        J = 0
        for subdir in os.listdir(self.dataDirectory):
            if J > self.num_patients:
                break
            data_dirs = os.listdir(self.dataDirectory + "/" + subdir)
            seg_image = nib.load(self.dataDirectory + "/" + subdir + "/" + [s for s in data_dirs if "seg" in s][0]).get_data()
            
            inds = [i for i in list(range(155)) if np.count_nonzero(seg_image[:,:,i]) > 0]
            foo = {}

            for mode in self.modes:
                for path in data_dirs:
                    if mode + ".nii" in path:
                        image = nib.load(self.dataDirectory + "/" + subdir + "/" + path).get_fdata()
                        foo[mode] = image
            for i in inds:
                j = 0
                augment_list = []
                img = np.zeros((self.W, self.H, len(self.modes)))
                for mode in self.modes:
                    proc_img, rmin, rmax, cmin, cmax = self.processImage(foo[mode][:,:,i])
                    #proc_img = self.windowIntensity(proc_img)
                    img[:,:,j] = proc_img
                    augment_list.append(img[:,:,j])
                    j = j+1
                
                seg_img = seg_image[:,:,i]
                seg_img[seg_img > 0] = 1
                seg_img = seg_img[rmin:rmax, cmin:cmax]
                seg_img = cv2.resize(seg_img, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
                augment_list.append(seg_img)
                
                if self.mode == "training":
                    num_augmentations = 1
                    for i in range(num_augmentations):
                        augmented_data = self.augmentData(augment_list)
                        aug_images = np.squeeze(np.array(augmented_data[:-1]))
                        self.X.append(aug_images)
                        """
                        
                        fig = plt.figure()
                        plt.gray();   
                        
                        fig.add_subplot(2,2,1)
                        plt.imshow(img[:,:,0])
                        plt.axis('off')
                        plt.title('Original FLAIR')
                        
                        fig.add_subplot(2,2,2)
                        plt.imshow(seg_img)
                        plt.axis('off')
                        plt.title('Original Segment')
                        
                        fig.add_subplot(2,2,3)
                        plt.imshow(augmented_data[0])
                        plt.axis('off')
                        plt.title('Augmented FLAIR')
                        
                        fig.add_subplot(2,2,4)
                        plt.imshow(augmented_data[1])
                        plt.axis('off')
                        plt.title('Augmented Segment')
                        
                        plt.show()
                        """
                        
                        aug_seg = augmented_data[-1]
                        self.labels.append(aug_seg)
                
                    #print(img.shape)
                    self.X.append(np.squeeze(img))
                    self.labels.append(seg_img)
                else:
                    seg_img = seg_img.reshape(seg_img.shape[0] * seg_img.shape[1])
                    #print(img.shape)
                    self.X.append(np.squeeze(img))
                    self.labels.append(seg_img)
    
            J = J+1

    def processData(self, image, seg_image, num_class = 2):
        #image = self.preprocess(image)
        rmin,rmax, cmin, cmax = self.bbox(image)
        
        image = image[rmin:rmax, cmin:cmax]
        #image = resize(image, (self.W, self.H))
        img = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        
        seg_image = seg_image[rmin:rmax, cmin:cmax]
        #seg_image = resize(seg_image, (self.W, self.H))
        mask = cv2.resize(seg_image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        flag_multi_class= False
        if(flag_multi_class):
            img = img / 255
            mask = mask.reshape((1,240,240))
            mask = mask[:,:,0]
            new_mask = np.zeros(mask.shape + (num_class,))
            for i in range(num_class):
                #for one pixel in the image, find the class in mask and convert it into one-hot vector
                #index = np.where(mask == i)
                #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
                #new_mask[index_mask] = 1
                new_mask[mask == i,i] = 1
            new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
            mask = new_mask
        elif(np.max(img) > 1):
            img = img / 255
            #mask = mask /255
            mask[mask > 0] = 1
            #mask[mask <= 0.5] = 0
        self.X.append(img)
        self.labels.append(mask)
      


    def getNumLabels(self):
        return self.labels[0].shape[1]
    
    def preprocessForNetwork(self):
        n_imgs = len(self.X)
        self.X = np.array( self.X )
        self.X = self.X.reshape(n_imgs,self.W, self.H,len(self.modes))
        self.labels = np.array( self.labels )
        self.labels = self.labels.reshape(n_imgs, self.W, self.H, 1)
       