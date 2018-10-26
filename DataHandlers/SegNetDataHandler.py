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
from dipy.segment.mask import multi_median, median_otsu
class SegNetDataHandler(DataHandler):
    modes = None
    mode = None # training, testing, or validation
    num_augments = 1

    def __init__(self,dataDirectory, 
                 W = 128, 
                 H = 128, 
                 num_patients = 3, 
                 modes = ["flair", "t1ce", "t1", "t2"]):
        super().__init__(dataDirectory, W, H, num_patients)
        self.modes = modes
        
    def windowIntensity(self, image, min_percent=1, max_percent=99):
        sitk_image = sitk.GetImageFromArray(image)
        sitk_image = sitk.Cast( sitk_image, sitk.sitkFloat32 )
        corrected_image = sitk.IntensityWindowing(sitk_image, 
                                                  np.percentile(image, min_percent), 
                                                  np.percentile(image, max_percent))
        corrected_image = sitk.GetArrayFromImage(corrected_image)
        return corrected_image
        

    def loadData(self):
        main_dir = os.listdir(self.dataDirectory)[0:self.num_patients+1]
        for subdir in main_dir:
            image_dir = self.dataDirectory + "/" + subdir
            data_dirs = os.listdir(image_dir)
            seg_image = nib.load(image_dir+
                                   "/" + 
                                   [s for s in data_dirs if "seg" in s][0]).get_fdata(caching = "unchanged",
                                                                                      dtype = np.float32)
            
            inds = [i for i in list(range(155)) if np.count_nonzero(seg_image[:,:,i]) > 0]
            foo = {}
            for mode in self.modes:
                for path in data_dirs:
                    if mode + ".nii" in path:
                        foo[mode] = nib.load(image_dir +
                                          "/" + 
                                          path).get_fdata(caching = "unchanged",
                                                          dtype = np.float32)
                        if len(foo) == len(self.modes):
                            break
            data = [self.resizeImages(foo, seg_image,i) for i in inds]
            train, labels = zip(*data)
            self.X.extend(train)
            self.labels.extend(labels)


                
                
    
    def resizeImages(self, foo, seg_image, i):
        img = np.zeros((self.W, self.H, len(self.modes)))
        for j,mode in enumerate(self.modes):
            img[:,:,j], rmin, rmax, cmin, cmax = self.processImage(foo[mode][:,:,i])
        
        seg_img = seg_image[:,:,i]
        seg_img = seg_img[rmin:rmax, cmin:cmax]
        seg_img = cv2.resize(seg_img, 
                     dsize=(self.W, self.H), 
                     interpolation=cv2.INTER_LINEAR)
        #seg_img[seg_img > 0] = 1
        return img, seg_img
                    
                        
    
    
    def processImage(self, image):
        rmin,rmax, cmin, cmax = self.bbox(image)
        image = image[rmin:rmax, cmin:cmax]
        resized_image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return resized_image, rmin, rmax, cmin, cmax
    
    def applyHMRF(self, img):
        hmrf = TissueClassifierHMRF(verbose=False)
        _, _, PVE = hmrf.classify(img, nclasses=3, beta=0.1)
        PVE = np.squeeze(PVE)
        white_matter = PVE[:,:,2:3]        
        csf = PVE[:,:,0:1]
        return img*white_matter
        #return img*white_matter# + img*csf
    
        """
        fig = plt.figure()
        
        a = fig.add_subplot(2, 3, 1)
        plt.imshow(np.squeeze(img), cmap="gray")
        a.set_title('Original')
        plt.axis('off')
        
        a = fig.add_subplot(2, 3, 2)
        plt.imshow(np.squeeze(med_img), cmap="gray")
        a.set_title('Median Otsu')
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
        plt.imshow(white_matter,cmap="gray")
        a.set_title('White Matter')
        plt.axis('off')
        
        plt.show()
        
        """
        
        
        
        #return np.squeeze(PVE)

    def preprocessData(self, img):
        ### preprocessing HMRF
        img = self.windowIntensity(img)
        #img = self.applyHMRF(img)
        return img

    
    def setMode(self, mode):
        self.mode = mode
    
    def getMode(self):
        return self.mode


    def getNumLabels(self):
        return self.labels[0].shape[1]
    
    def preprocessForNetwork(self):
        
        pool = Pool(processes=5)
        self.X = pool.map(self.preprocessData, self.X)
        pool.close()
        pool.join()
        """
        n_imgs = len(self.X)
        #self.X = np.array(self.X)
        
        # z-score norm
        sigma = np.std(self.X)
        mu = np.mean(self.X)
        self.X = (self.X - mu)/sigma
        
        
        self.X = self.X.reshape(n_imgs,self.W, self.H,len(self.modes))
        """
        #self.labels = [label.reshape(label.shape[0] * label.shape[1]) for label in self.labels]
        #self.labels = np.array( self.labels )
