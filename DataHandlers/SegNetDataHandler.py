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
from skimage import exposure
class SegNetDataHandler(DataHandler):
    modes = None

    def __init__(self,dataDirectory, W = 128, H = 128, num_patients = 3, modes = ["flair", "t1ce", "t1", "t2"]):
        super().__init__(dataDirectory, W, H, num_patients)
        self.modes = modes
        
    def windowIntensity(self, image, min_percent=1, max_percent=99):
        sitk_image = sitk.GetImageFromArray(image)
        sitk_image = sitk.Cast( sitk_image, sitk.sitkFloat64 )
        corrected_image = sitk.IntensityWindowing(sitk_image, np.percentile(image, min_percent), 
                                                  np.percentile(image, max_percent))
        corrected_image = sitk.GetArrayFromImage(corrected_image)
        corrected_image = exposure.equalize_hist(corrected_image)
        return corrected_image
        

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
                    proc_img = self.windowIntensity(proc_img)
                    #img[:,:,j] = foo[mode][:,:,i]
                    img[:,:,j] = proc_img
                    j = j+1
                self.X.append(img)
                
                seg_img = seg_image[:,:,i]
                seg_img[seg_img > 0] = 1
                seg_img = seg_img[rmin:rmax, cmin:cmax]
                seg_img = cv2.resize(seg_img, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
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
                    
                        
                        
    def processImage(self, image):
        rmin,rmax, cmin, cmax = self.bbox(image)
        image = image[rmin:rmax, cmin:cmax]
        resized_image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return resized_image, rmin, rmax, cmin, cmax
        
        

    def processData(self, image, seg_image):
        seg_image = seg_image.reshape(seg_image.shape[0] * seg_image.shape[1])
        self.X.append(image)
        self.labels.append(seg_image)

    def getNumLabels(self):
        return self.labels[0].shape[1]
    
    def preprocessForNetwork(self):
        n_imgs = len(self.X)
        self.X = np.array( self.X )
        self.X = self.X / 255
        self.X = self.X.reshape(n_imgs,self.W, self.H,len(self.modes))
        self.labels = np.array( self.labels )
