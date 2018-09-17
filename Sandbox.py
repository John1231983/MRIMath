'''
Created on Jun 25, 2018

@author: daniel
'''
from __future__ import print_function

import os
from multiprocessing import Pool, cpu_count
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import skimage.color
from NMFComputer import NMFComputer
from Utils.TimerModule import TimerModule
from functools import partial
from skimage import exposure
import numpy as np

timer = TimerModule()
def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

# simple bias field correction
def processImage(image, i):
    sitk_image = sitk.GetImageFromArray(image[:,:,i])
    sitk_image = sitk.Cast( sitk_image, sitk.sitkFloat64 )
    corrected_image = sitk.N4BiasFieldCorrection(sitk_image, sitk_image > 0);
    corrected_image = sitk.GetArrayFromImage(corrected_image)
    return corrected_image
        


def removeBiasField(image):
    
    sitk_image = sitk.GetImageFromArray(image)
    #sitk_image = sitk.IntensityWindowing(sitk_image, np.percentile(image, 1), np.percentile(image, 99))
    sitk_image = sitk.Cast( sitk_image, sitk.sitkFloat64 )

    corrected_image = sitk.N4BiasFieldCorrection(sitk_image, sitk_image > 0);
    corrected_image = sitk.GetArrayFromImage(corrected_image)
    #corrected_image = exposure.equalize_hist(corrected_image)
    return corrected_image
        
def windowIntensity(image, min_percent=1, max_percent=99):
    
    sitk_image = sitk.GetImageFromArray(image)
    #sitk_image = sitk.IntensityWindowing(sitk_image, np.percentile(image, 1), np.percentile(image, 99))
    sitk_image = sitk.Cast( sitk_image, sitk.sitkFloat64 )
    corrected_image = sitk.IntensityWindowing(sitk_image, np.percentile(image, min_percent), np.percentile(image, max_percent))

    corrected_image = sitk.GetArrayFromImage(corrected_image)
    #corrected_image = exposure.equalize_hist(corrected_image)
    return corrected_image

def rescaleIntensity(image, minimum=0, maximum=20000):
    
    sitk_image = sitk.GetImageFromArray(image)
    #sitk_image = sitk.IntensityWindowing(sitk_image, np.percentile(image, 1), np.percentile(image, 99))
    sitk_image = sitk.Cast( sitk_image, sitk.sitkFloat64 )
    corrected_image = sitk.RescaleIntensity(sitk_image, minimum, maximum)
    corrected_image = sitk.GetArrayFromImage(corrected_image)
    #corrected_image = exposure.equalize_hist(corrected_image)
    return corrected_image        

def main():
    
    directory_with_data = ""
    # don't do this 
    
    

    
    data_dir = "Data/BRATS_2018/HGG_Validation"
    for subdir in os.listdir(data_dir):
        for path in os.listdir(data_dir + "/" + subdir):
            if "flair.nii" in path:
                image = nib.load(data_dir + "/" + subdir + "/" + path)
                seg_image = nib.load(data_dir + "/" + subdir + "/" + path.replace("flair", "seg"))
                
                

    
        fig = plt.figure()
        plt.gray()

        image = image.get_fdata()[:,:,100]
        seg_image = seg_image.get_fdata()[:,:,100]
                
        a=fig.add_subplot(1,6,1)
        plt.imshow(image)
        a.set_title('Original Image')
        plt.axis('off')

            
        bf_corrected = removeBiasField(image)
        windowed = windowIntensity(image)
        a=fig.add_subplot(1,6,2)
        #image_temp = exposure.equalize_hist(image_temp)
        plt.imshow(bf_corrected)
        a.set_title('Bias Field Corrected')
        plt.axis('off')
        
        windowed = windowIntensity(image)
        a=fig.add_subplot(1,6,3)
        #image_temp = exposure.equalize_hist(image_temp)
        plt.imshow(windowed)
        a.set_title('Intensity Windowed')
        plt.axis('off')

        combined = removeBiasField(windowIntensity(image))
        a=fig.add_subplot(1,6,4)
        #image_temp = exposure.equalize_hist(image_temp)
        plt.imshow(combined)
        a.set_title('Combined (bf + window)')
        plt.axis('off')
        
        
        combined2 = windowIntensity(removeBiasField(image))
        a=fig.add_subplot(1,6,5)
        #image_temp = exposure.equalize_hist(image_temp)
        plt.imshow(combined2)
        a.set_title('Combined (window + bf)')
        plt.axis('off')
        
        a=fig.add_subplot(1,6,6)
        #image_temp = exposure.equalize_hist(image_temp)
        plt.imshow(seg_image)
        a.set_title('GT Segment')
        plt.axis('off')
        plt.show()
            
            

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()