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

from skimage import exposure
import numpy as np
def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)







def main():
    temp_dir = "Data/BRATS_2018/HGG_BF"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    data_dir = "Data/BRATS_2018/HGG"
    for subdir in os.listdir(data_dir):
        for path in os.listdir(data_dir + "/" + subdir):
            if "t1ce" in path:
                image = nib.load(data_dir + "/" + subdir + "/" + path)
                break
        fig = plt.figure()

        image = image.get_data()[:,:,1]
        a=fig.add_subplot(1,2,1)
        imgplot = plt.imshow(image)
        a.set_title('Before')
            
        sitk_image = sitk.GetImageFromArray(image)
        #sitk_image = sitk.IntensityWindowing(sitk_image, np.percentile(image, 1), np.percentile(image, 99))

        sitk_image = sitk.Cast( sitk_image, sitk.sitkFloat64 )

        image_temp = sitk.N4BiasFieldCorrection(sitk_image, sitk_image > 0);
        image_temp = sitk.GetArrayFromImage(image_temp)
        image_temp = image_temp.astype(np.uint8)

        image_temp = cv2.cvtColor(image_temp,cv2.COLOR_GRAY2RGB)

        a=fig.add_subplot(1,2,2)
        #image_temp = exposure.equalize_hist(image_temp)
        imgplot = plt.imshow(image_temp)
        a.set_title('After')
        plt.show()
            
            

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()