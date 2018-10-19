'''
Created on Aug 1, 2018

@author: daniel
'''
from DataHandlers.DataHandler import DataHandler
import numpy as np
import cv2
from keras.utils import np_utils
import os
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.transform import resize
import SimpleITK as sitk
from dipy.segment.tissue import TissueClassifierHMRF
from multiprocessing import Pool

class SegNetDataHandler(DataHandler):
    modes = None
    mode = None # training, testing, or validation
    num_augments = 2

    def __init__(self,dataDirectory, W = 128, H = 128, num_patients = 3, modes = ["flair", "t1ce", "t1", "t2"]):
        super().__init__(dataDirectory, W, H, num_patients)
        self.modes = modes
        
    def windowIntensity(self, image, min_percent=1, max_percent=99):
        sitk_image = sitk.GetImageFromArray(image)
        sitk_image = sitk.Cast( sitk_image, sitk.sitkFloat64 )
        corrected_image = sitk.IntensityWindowing(sitk_image, 
                                                  np.percentile(image, min_percent), 
                                                  np.percentile(image, max_percent))
        corrected_image = sitk.GetArrayFromImage(corrected_image)
        return corrected_image
        

    def loadData(self):
        main_dir = os.listdir(self.dataDirectory)[0:self.num_patients+1]
        for subdir in main_dir:
            data_dirs = os.listdir(self.dataDirectory +
                                    "/" +
                                     subdir)
            seg_image = nib.load(self.dataDirectory + 
                                 "/" +
                                  subdir +
                                   "/" + 
                                   [s for s in data_dirs if "seg" in s][0]).get_fdata(caching = "unchanged",
                                                                                      dtype = np.float32)
            
            inds = [i for i in list(range(155)) if np.count_nonzero(seg_image[:,:,i]) > 0]
            foo = {}

            for mode in self.modes:
                for path in data_dirs:
                    if mode + ".nii" in path:
                        image = nib.load(self.dataDirectory + 
                                         "/" + 
                                         subdir +
                                          "/" + 
                                          path).get_fdata(caching = "unchanged",
                                                          dtype = np.float32)
                        foo[mode] = image
            for i in inds:
                img = np.zeros((self.W, self.H, len(self.modes)))
                for j,mode in enumerate(self.modes):
                    proc_img, rmin, rmax, cmin, cmax = self.processImage(foo[mode][:,:,i])
                    img[:,:,j] = self.windowIntensity(proc_img)
                    
                seg_img = seg_image[:,:,i]
                seg_img = seg_img[rmin:rmax, cmin:cmax]
                seg_img = cv2.resize(seg_img, 
                                     dsize=(self.W, self.H), 
                                     interpolation=cv2.INTER_LINEAR)
                seg_img[seg_img > 0] = 1


                self.X.append(img)
                self.labels.append(seg_img)


                
                
    
    def resizeWindowAndProcess(self, foo, seg_image, i):
        img = np.zeros((self.W, self.H, len(self.modes)))
        for j,mode in enumerate(self.modes):
            proc_img, rmin, rmax, cmin, cmax = self.processImage(foo[mode][:,:,i])
            img[:,:,j] = self.windowIntensity(proc_img)
        
        hmrf = self.hmrf(img)
        img *= hmrf[:,:,2]
        seg_img = seg_image[:,:,i]
        seg_img = seg_img[rmin:rmax, cmin:cmax]
        seg_img = cv2.resize(seg_img, 
                     dsize=(self.W, self.H), 
                     interpolation=cv2.INTER_LINEAR)
        seg_img[seg_img > 0] = 1
        return np.squeeze(img), seg_img
                    
                        
    def augmentImages(self, ind):
        for _ in range(self.num_augments):
            augmented_data = self.augmentData([self.X[ind], self.labels[ind]])
            aug_images = np.squeeze(np.array(augmented_data[:-1]))
            self.X.append(aug_images)
            aug_seg = augmented_data[-1]
            self.labels.append(aug_seg)
                
            """
            fig = plt.figure()
            plt.gray();   
            
            fig.add_subplot(2,2,1)
            plt.imshow(self.X[ind][:,:])
            plt.axis('off')
            plt.title('Original FLAIR')
            
            fig.add_subplot(2,2,2)
            plt.imshow(self.labels[ind])
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
                           
    def processImage(self, image):
        rmin,rmax, cmin, cmax = self.bbox(image)
        image = image[rmin:rmax, cmin:cmax]
        resized_image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return resized_image, rmin, rmax, cmin, cmax
    
    def hmrf(self, img):
        nclass = 3
        beta = 0.2
        hmrf = TissueClassifierHMRF(verbose=False)
        _, _, PVE = hmrf.classify(img, nclass, beta)
        return np.squeeze(PVE)
        """
        fig = plt.figure()
        
        a = fig.add_subplot(2, 3, 1)
        plt.imshow(np.squeeze(img), cmap="gray")
        a.set_title('Original')
        plt.axis('off')

        a = fig.add_subplot(2, 3, 2)
        plt.imshow(np.squeeze(final_segmentation), cmap="gray")
        a.set_title('Processed')
        plt.axis('off')
        
        a = fig.add_subplot(2, 3, 3)
        plt.imshow(np.squeeze(seg_img),cmap="gray")
        a.set_title('GT Segmentation')
        plt.axis('off')
        
        PVE = np.squeeze(PVE)
        a = fig.add_subplot(2, 3, 4)
        plt.imshow(PVE[:,:,0], cmap="gray")
        a.set_title('CSF')
        plt.axis('off')

        a = fig.add_subplot(2, 3, 5)
        plt.imshow(PVE[:,:,1], cmap="gray")
        a.set_title('Gray Matter')
        plt.axis('off')
        
        a = fig.add_subplot(2, 3, 6)
        plt.imshow(PVE[:,:,2],cmap="gray")
        a.set_title('White Matter')
        plt.axis('off')
        
        plt.show()
        """
    
    def setMode(self, mode):
        self.mode = mode
    
    def getMode(self):
        return self.mode

    def processData(self, image, seg_image):
        seg_image = seg_image.reshape(seg_image.shape[0] * seg_image.shape[1])
        self.X.append(image)
        self.labels.append(seg_image)

    def getNumLabels(self):
        return self.labels[0].shape[1]
    
    def preprocessForNetwork(self):
        n_imgs = len(self.X)
        ### preprocessing HMRF
        
        pool = Pool(processes=7)
        hmrf = pool.map(self.hmrf, self.X)
        for i in range(len(self.X)):
            foo = hmrf[i]
            self.X[i] = self.X[i]*foo[:,:,2:3] + self.X[i]*foo[:,:,0:1]
        
        self.X = np.array( self.X )
        self.X = self.X.reshape(n_imgs,self.W, self.H,len(self.modes))
        self.labels = [label.reshape(label.shape[0] * label.shape[1]) for label in self.labels]
        self.labels = np.array( self.labels )
