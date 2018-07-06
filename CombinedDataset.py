'''
Created on Jun 30, 2018

@author: daniel
'''
from MRIMathDataset import MRIMathDataset
import os
import skimage.color
import nibabel as nib
from mrcnn import utils
import numpy as np

class CombinedDataset(MRIMathDataset):
    
    def load_images(self, data_dir):
        print('Reading images')
        # Add classes
        self.add_class("mrimath", 1, "whole")
        self.add_class("mrimath", 2, "active")
        self.add_class("mrimath", 3, "core")


        i = 0
        for subdir in os.listdir(data_dir):
            indices = self.getIndicesWithTumorPresent(data_dir + "/" + subdir)
            for j in indices:
                self.add_image("mrimath", image_id=i, path=data_dir + "/" + subdir, ind = j)
                i = i + 1
    def load_image(self, image_id):
        ## Note:
        # FLAIR -> Whole
        # T2 -> Core
        # T1C -> Active (if present)
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        image = np.zeros((240,240,3))
        info = self.image_info[image_id]
        for path in os.listdir(info['path']):
            if "flair" in path:
                image[:,:,0] = nib.load(info['path'] + "/" + path).get_data()[:,:,info['ind']]
            elif "t1ce" in path:
                image[:,:,1] = nib.load(info['path'] + "/" + path).get_data()[:,:,info['ind']]
                #image[:,:,1] = self.preprocess_image(image[:,:,1])
            elif "t2" in path:
                image[:,:,2] = nib.load(info['path'] + "/" + path).get_data()[:,:,info['ind']]
                #image[:,:,2] = self.preprocess_image(image[:,:,2])
        image = self.preprocess_image(image)
        return image
    def load_mask(self, image_id):
      
        for path in os.listdir(self.image_info[image_id]['path']):
            if "seg" in path:
                mask = nib.load(self.image_info[image_id]['path']+"/"+path).get_data()[:,:,self.image_info[image_id]['ind']]
                break
        """
        plt.figure(1)
        plt.imshow(mask)
        plt.show()
        """
        mask, class_ids = self.getMask(mask)
        return mask.astype(bool), np.asarray(class_ids, dtype=np.int32)
        
    def getMask(self, mask):
        a = []
        class_ids = []
        
        
        whole = self.getWholeMask(mask.copy())
        if np.count_nonzero(whole) > 0:
            class_ids.append(1)
            a.append(whole)
            
            
        active = self.getActiveMask(mask.copy())
        if np.count_nonzero(active) > 0:
            class_ids.append(2)
            a.append(active)

            
        core = self.getCoreMask(mask.copy())
        if np.count_nonzero(core) > 0:
            class_ids.append(3)
            a.append(core)
            
        temp = np.array(a)
        temp = np.swapaxes(temp, 0, 2)
        return temp, class_ids
      
    def getWholeMask(self, mask):
        mask[mask > 0] = 1
        return mask
    
    def getActiveMask(self, mask):
        mask[mask < 4] = 0
        mask[mask > 0] = 1
        return mask
        
    def getCoreMask(self, mask):
        mask[mask == 2] = 0
        mask[mask > 0] = 1
        return mask
        
    
        