import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import shutil
import random
# Root directory of the project
ROOT_DIR = os.path.abspath("Mask_RCNN")
print(ROOT_DIR)
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from multiprocessing import Pool, Manager, Process, Lock

import skimage.color

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class MRIMathConfig(Config):
    # Give the configuration a recognizable name
    NAME = "mrimath"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 2 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    #IMAGE_MIN_DIM = 256
    
    #IMAGE_MAX_DIM = 256
    #LEARNING_RATE = 0.0001
    #LEARNING_RATE = 0.00001

    # Use smaller anchors because our image and objects are small
    #RPN_ANCHOR_SCALES = (16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    #TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    #STEPS_PER_EPOCH = 120
    #BACKBONE = "resnet50"


    # use small validation steps since the epoch is small
    #VALIDATION_STEPS = 30
    
class MRIMathDataset(utils.Dataset):
    mode = None
    tumor_type = None
    

            
    def load_image(self, image_id):
        ## Note:
        # FLAIR -> Whole
        # T2 -> Core
        # T1C -> Active (if present)
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        info = self.image_info[image_id]
        for path in os.listdir(info['path']):
            if self.mode in path:
                image = nib.load(info['path'] + "/" + path).get_data()[:,:,info['ind']]
                break;
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image
    
    def load_images(self, data_dir):
        print('Reading images')
        # Add classes
        self.add_class("mrimath", 1, self.tumor_type)
        i = 0
        for subdir in os.listdir(data_dir):
            for j in range(0,155):
                self.add_image("mrimath", image_id=i, path=data_dir + "/" + subdir, ind = j)
                if self.checkIfTumorPresent(i): 
                    i = i + 1
                    
    def checkIfTumorPresent(self, image_id):
        info = self.image_info[image_id]
        path = next((s for s in os.listdir(info['path']) if "seg" in s), None)
        if "seg" in path:
            mask = nib.load(info['path']+"/"+path).get_data()[:,:,info['ind']]
            if np.count_nonzero(mask) <= 0:
                self.image_info.remove(info)
                return False
        return True
        
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "mrimath":
            return info["source"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """        
        for path in os.listdir(self.image_info[image_id]['path']):
            if "seg" in path:
                mask = nib.load(self.image_info[image_id]['path']+"/"+path).get_data()[:,:,self.image_info[image_id]['ind']]
                break
        """
        plt.figure(1)
        plt.imshow(mask)
        plt.show()
        """
        mask = self.getMask(mask)
        mask = mask.reshape(mask.shape[0], mask.shape[1],1)
        return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def getMask(self, mask):
        pass

class FlairDataset(MRIMathDataset):

    mode = "flair"
    tumor_type = "whole"
    
    def getMask(self, mask):
        mask[mask > 0] = 1
        return mask

class T2Dataset(MRIMathDataset):
        
    def __init__(self):
        super().__init__()

        self.mode = "t2"
        self.tumor_type = "core"
    
    def getMask(self, mask):
        
        mask[mask == 2] = 0
        mask[mask > 0] = 1
        return mask

class T1CDataset(MRIMathDataset):
    def __init__(self):
        super().__init__()
        self.mode = "t1ce"
        self.tumor_type = "active"
    
    def getMask(self, mask):
        mask[mask < 4] = 0
        mask[mask > 0] = 1
        return mask

class InferenceConfig(MRIMathConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# stuff to run always here such as class/def
def main():
    config = MRIMathConfig()
    config.display()
    
    random.seed(12345)
    data_dir = "Data/BRATS_2018/HGG"
    val_dir = "Data/BRATS_2018/HGG_Validation"
    test_dir = "Data/BRATS_2018/HGG_Testing"
    
    if os.listdir(val_dir) == []:
        # split validation data (10% of dataset)
        list_imgs = os.listdir(data_dir)
        val_imgs = random.sample(list_imgs, round(0.05*len(list_imgs)))
        for sub_dir in list_imgs:
            if sub_dir in val_imgs:
                dir_to_move = os.path.join(data_dir, sub_dir)
                shutil.move(dir_to_move, val_dir)
                
    if os.listdir(test_dir) == []:
        # split testing data (20% of dataset)
        list_imgs = os.listdir(data_dir)
        test_imgs = random.sample(list_imgs, round(0.05*len(list_imgs)))
        for sub_dir in list_imgs:
            if sub_dir in test_imgs:
                dir_to_move = os.path.join(data_dir, sub_dir)
                shutil.move(dir_to_move, test_dir)
            
    dataset_train = FlairDataset()
    dataset_train.load_images(data_dir)
    dataset_train.prepare()
    
    
    dataset_val = FlairDataset()
    dataset_val.load_images(val_dir)
    dataset_val.prepare()
    
    print("Training on " + str(len(dataset_train.image_info)) + " images")
    print("Validating on " + str(len(dataset_val.image_info)) + " images")


        # Validation dataset
    #dataset_val = MRIMathDataset()
    #dataset_val.load_images( '/media/daniel/Backup Data/Flair', 130,180)
    #dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last
    
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)
    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=500,
                layers='heads')

    # move the validation data back
    list_imgs = os.listdir(val_dir)
    for sub_dir in list_imgs:
        dir_to_move = os.path.join(val_dir, sub_dir)
        shutil.move(dir_to_move, data_dir)
        
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
    
