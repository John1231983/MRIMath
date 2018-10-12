'''
Created on Aug 29, 2018

@author: daniel
'''

from Exploratory_Stuff.DataHandler import DataHandler
import numpy as np
import cv2
from keras.utils import np_utils
import os
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.transform import resize
import SimpleITK as sitk
class UNetDataHandler(DataHandler):
    modes = None
    def __init__(self,dataDirectory, nmfComp, W = 128, H = 128, num_patients = 3, modes = ["flair", "t1ce", "t1", "t2"]):
        super().__init__(dataDirectory, nmfComp, W, H, num_patients)
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
                img = np.zeros((self.W, self.H, len(self.modes)))
                for mode in self.modes:
                    proc_img, rmin, rmax, cmin, cmax = self.processImage(foo[mode][:,:,i])
                    #img[:,:,j] = foo[mode][:,:,i]
                    img[:,:,j] = proc_img
                    j = j+1
                self.X.append(img)
                
                seg_img = seg_image[:,:,i]
                seg_img[seg_img > 0] = 1
                seg_img = seg_img[rmin:rmax, cmin:cmax]
                seg_img = cv2.resize(seg_img, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
                #proc_img = exposure.equalize_hist(proc_img)
                """
                fig = plt.figure()
                plt.gray();   
                
                fig.add_subplot(1,3,1)
                plt.imshow(img[:,:,j-1])
                plt.axis('off')
                plt.title('Original')
                
                fig.add_subplot(1,3,2)
                plt.imshow(window_proc_img)
                plt.axis('off')
                plt.title('Windowed')
                
                
                fig.add_subplot(1,3,3)
                plt.imshow(seg_img)
                plt.axis('off')
                plt.title('Segment')
                plt.show()
                """
                seg_img = seg_img.reshape(seg_img.shape[0] * seg_img.shape[1])
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
        """
        rmin,rmax, cmin, cmax = self.bbox(image)
        
        image = image[rmin:rmax, cmin:cmax]
        #image = resize(image, (self.W, self.H))
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        
        seg_image = seg_image[rmin:rmax, cmin:cmax]
        #seg_image = resize(seg_image, (self.W, self.H))
        seg_image = cv2.resize(seg_image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)

        """

        """
        fig = plt.figure()
        plt.gray();   
        
        a=fig.add_subplot(1,2,1)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Original')
        
        a=fig.add_subplot(1,2,2)
        plt.imshow(seg_image)
        plt.axis('off')
        plt.title('Segment')
        plt.show()
        """
        
        
        
        """
        indices = np.argmax(W, axis=0)
        #H = H[indices > 0]
        regions = np.argmax(H, axis=0)
        
        
        
        region_1 = regions.copy()
        region_1_and_2 = regions.copy()
        region_1_and_2_and_3 = regions.copy()
        region_1_and_2_and_3_and_4 = regions.copy()
        region_1_and_2_and_3_and_4_and_5 = regions.copy()
        region_1_and_2_and_3_and_4_and_5_and_6 = regions.copy() 
        
        region_1[regions < 0] = 0
        region_1[regions > 1] = 0
        region_1 = region_1.astype(bool)
        
        region_1_and_2[regions < 1] = 0
        region_1_and_2[regions > 2] = 0
        region_1_and_2 = region_1_and_2.astype(bool)
                
        region_1_and_2_and_3[regions < 1] = 0
        region_1_and_2_and_3[regions > 3] = 0
        region_1_and_2_and_3 = region_1_and_2_and_3.astype(bool)
        
        region_1_and_2_and_3_and_4[regions < 1] = 0
        region_1_and_2_and_3_and_4[regions > 4] = 0
        region_1_and_2_and_3_and_4 = region_1_and_2_and_3_and_4.astype(bool)
        
        region_1_and_2_and_3_and_4_and_5[regions < 1] = 0
        region_1_and_2_and_3_and_4_and_5[regions > 5] = 0
        region_1_and_2_and_3_and_4_and_5 = region_1_and_2_and_3_and_4_and_5.astype(bool)

        
        region_1_and_2_and_3_and_4_and_5_and_6[regions < 1] = 0
        region_1_and_2_and_3_and_4_and_5_and_6[regions > 6] = 0
        region_1_and_2_and_3_and_4_and_5_and_6 = region_1_and_2_and_3_and_4_and_5_and_6.astype(bool)
        
        reg_1_image = image.copy()
        reg_1_and_2_image = image.copy()
        reg_1_and_2_and_3_image = image.copy()
        reg_1_and_2_and_3_and_4_image = image.copy()
        reg_1_and_2_and_3_and_4_and_5_image = image.copy()
        reg_1_and_2_and_3_and_4_and_5_and_6_image = image.copy()
        
        #regions = regions.astype(bool)
        m = self.nmfComp.block_dim
        ind = 0
        for i in range(0, seg_image.shape[0], m):
            for j in range(0, seg_image.shape[1], m):
                reg_1_image[i:i+m, j:j+m] *= region_1[ind]
                reg_1_and_2_image[i:i+m, j:j+m] *= region_1_and_2[ind]
                reg_1_and_2_and_3_image[i:i+m, j:j+m] *= region_1_and_2_and_3[ind]
                reg_1_and_2_and_3_and_4_image[i:i+m, j:j+m] *= region_1_and_2_and_3_and_4[ind]
                reg_1_and_2_and_3_and_4_and_5_image[i:i+m, j:j+m] *= region_1_and_2_and_3_and_4_and_5[ind] 
                reg_1_and_2_and_3_and_4_and_5_and_6_image[i:i+m, j:j+m] *= region_1_and_2_and_3_and_4_and_5_and_6[ind]
                ind = ind + 1
                """
        """
        
        dims = tuple([self.W, self.W])
        seg_image[seg_image > 0] = 1
        seg_image = self.label_map(seg_image, 1)
        
        X.append(image)
        y.append(seg_image)
        
        reg_1_and_2_and_3_image = cv2.resize(reg_1_and_2_and_3_image,dims)
        X.append(reg_1_and_2_and_3_image)
        y.append(seg_image)
        
        reg_1_and_2_and_3_and_4_image = cv2.resize(reg_1_and_2_and_3_and_4_image,dims)
        X.append(reg_1_and_2_and_3_and_4_image)
        y.append(seg_image)
        
        reg_1_and_2_and_3_and_4_and_5_image = cv2.resize(reg_1_and_2_and_3_and_4_and_5_image,dims)
        X.append(reg_1_and_2_and_3_and_4_and_5_image)
        y.append(seg_image)

        reg_1_and_2_and_3_and_4_and_5_and_6_image = cv2.resize(reg_1_and_2_and_3_and_4_and_5_and_6_image,dims)
        X.append(reg_1_and_2_and_3_and_4_and_5_and_6_image)
        y.append(seg_image)
        
        
        
        fig = plt.figure()
        plt.gray();   
        
        a=fig.add_subplot(1,8,1)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Original')
        
        a=fig.add_subplot(1,8,2)
        plt.imshow(reg_1_image)
        plt.axis('off')
        plt.title('1')
        
        
        a=fig.add_subplot(1,8,3)
        plt.imshow(reg_1_and_2_image)
        plt.axis('off')
        plt.title(r'1 $\cup$ 2')
        
        a=fig.add_subplot(1,8,4)
        plt.imshow(reg_1_and_2_and_3_image)
        plt.axis('off')
        plt.title(r'1 $\cup$ 2 $\cup$ 3')
        self.X.append(reg_1_and_2_and_3_image)
        
        
        a=fig.add_subplot(1,8,5)
        plt.imshow(reg_1_and_2_and_3_and_4_image)
        plt.axis('off')
        plt.title(r'1 $\cup$ 2 $\cup$ 3 $\cup$ 4')
        
        a=fig.add_subplot(1,8,6)
        plt.imshow(reg_1_and_2_and_3_and_4_and_5_image)
        plt.axis('off')
        plt.title(r'1 $\cup$ 2 $\cup$ 3 $\cup$ 4 $\cup$ 5')


        a=fig.add_subplot(1,8,7)
        plt.imshow(reg_1_and_2_and_3_and_4_and_5_and_6_image)
        plt.axis('off') 
        plt.title(r'1 $\cup$ 2 $\cup$ 3 $\cup$ 4 $\cup$ 5 $\cup$ 6')

        a=fig.add_subplot(1,8,8)
        plt.imshow(seg_image)
        plt.axis('off')
        plt.title('Segment')

        plt.show()
        """
        
        #return X, y


    def getNumLabels(self):
        return self.labels[0].shape[1]
    
    def preprocessForNetwork(self):
        n_imgs = len(self.X)
        self.X = np.array( self.X )
        self.X = self.X.reshape(n_imgs,self.W, self.H,len(self.modes))
        self.labels = np.array( self.labels )
        self.labels = self.labels.reshape(n_imgs, self.W, self.H, 1)
        #self.labels = np_utils.to_categorical(self.labels)        
        #self.labels = np.clip(self.labels, 0, 1)
        # self.labels = self.labels.reshape(n_imgs, self.W*self.H,2)