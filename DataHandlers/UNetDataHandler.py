'''
Created on Aug 1, 2018

@author: daniel
'''
from DataHandlers.DataHandler import DataHandler
import numpy as np
import cv2
import os
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
class UNetDataHandler(DataHandler):
    modes = None
    num_augments = 1

    def __init__(self,dataDirectory, 
                 W = 240, 
                 H = 240, 
                 num_patients = 3, 
                 modes = ["flair", "t1ce", "t1", "t2"],
                 n_labels = 1):
        super().__init__(dataDirectory, W, H, num_patients)
        self.modes = modes
        self.n_labels = n_labels
        
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
            data = [self.processImages(foo, seg_image,i) for i in inds]
            train, labels = zip(*data)
            self.X.extend(train)
            self.labels.extend(labels)


    
    def processImages(self, foo, seg_image, i):
        img = np.zeros((self.W, self.H, len(self.modes)))
        for j,mode in enumerate(self.modes):
            #img[:,:,j], rmin, rmax, cmin, cmax = self.zoomOnImage(foo[mode][:,:,i])
            img[:,:,j] = foo[mode][:,:,i]

        #img = self.windowIntensity(img)
        seg_img = seg_image[:,:,i]
        
        """
        seg_img = seg_img[rmin:rmax, cmin:cmax]
        seg_img = cv2.resize(seg_img, 
                     dsize=(self.W, self.H), 
                     interpolation=cv2.INTER_LINEAR)
        
        seg_img = np.rint(seg_img)
        
        """
        
        regions = np.zeros((seg_img.shape[0], seg_img.shape[1], self.n_labels))
        

        plt.show()
        if self.n_labels >= 1:
            region_1 = seg_img.copy()
            region_1[region_1 > 0] = 1
            regions[:,:,0] = region_1
        
        if self.n_labels >= 2:
            region_2 = seg_img.copy()
            region_2[region_2 == 2] = 0
            region_2[region_2 > 0] = 1
            regions[:,:,1] = region_2
        
        if self.n_labels >= 3:
            region_3 = seg_img.copy()
            region_3[region_3 < 4] = 0
            region_3[region_3 > 0] = 1
            regions[:,:,2] = region_3
        #self.showData(img[:,:,0], regions, seg_img)

        
        
        #print(regions.shape)
        return img, regions
                    
                        
    
    def showData(self, img, regions, seg_img):
        fig = plt.figure()
        plt.gray();   
        fig.add_subplot(1,self.n_labels+2,1)
        
        plt.imshow(img)
        plt.axis('off')
        plt.title('FLAIR')
        for i in range(1, self.n_labels+1):
            fig.add_subplot(1,self.n_labels+2,i+1)
            plt.imshow(regions[:,:,i-1])
            plt.axis('off')
            plt.title("Region " + str(i))
        
        
        fig.add_subplot(1,self.n_labels+2,self.n_labels+2)
        plt.imshow(seg_img)
        plt.axis('off')
        plt.title('GT Segment')
        

        plt.show()
        

    def zoomOnImage(self, image):
        rmin,rmax, cmin, cmax = self.bbox(image)
        image = image[rmin:rmax, cmin:cmax]
        resized_image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return resized_image, rmin, rmax, cmin, cmax
    


    
    def setMode(self, mode):
        self.mode = mode
    
    def getMode(self):
        return self.mode


    def getNumLabels(self):
        return self.labels[0].shape[1]
    
    def preprocessForNetwork(self):
        n_imgs = len(self.X)
        for label in self.labels:
            label[label > 0] = 1
        self.labels = [label.reshape(label.shape[0] * label.shape[1]) for label in self.labels]
        self.X = np.array(self.X)
        self.labels = np.array(self.labels)
        # z-score norm
        sigma = np.std(self.X)
        mu = np.mean(self.X)
        self.X = (self.X - mu)/sigma
        self.X = self.X.reshape(n_imgs,self.W, self.H,len(self.modes))
 