import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/daniel/Mask_RCNN")
print(ROOT_DIR)
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class MRIMathConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "mrimath"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16,32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 8

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
class MRIMathDataset(utils.Dataset):
        ## Reads an image from a filepath
    #
    # @param path the path to an image file
    # @return the image from the filepath (if one existed) as a numpy array
    def getImage(self, path):
        path=path.decode()
        img = cv2.imread(path,0)
        return img
    ## Constructs the patient directory string based on the index (based on current labeling scheme)
    #
    # @param index the index of the patient that you need the specific directory for
    # @param data_directory Directory where all patient data is located
    def getDirectoryFromIndex(self, index, data_directory):

        return data_directory + self.getPatientDirectoryFromIndex(index)
    
        ## Constructs the patient directory string based on the index (based on current labeling scheme)
    #
    # @param index the index of the patient that you need the specific directory for
    # @param data_directory Directory where all patient data is located
    def getPatientDirectoryFromIndex(self, index):
        if index < 10:
            patient_directory =  '/Patient_(00' + str(index)  + ')_data/'
        elif index < 100:
            patient_directory = '/Patient_(0' + str(index)  + ')_data/'
        else:
            patient_directory =  '/Patient_(' + str(index)  + ')_data/'
        return patient_directory
    ## Derives and labels patches from an individual patient
    #
    # @param data_directory the directory where all patient data is located
    # @param index index of the patient in the numbered directory
    def loadIndividualPatient(self, index, data_directory):
        print('Reading Patient ' + str(index))
        patient_directory = os.fsencode(self.getDirectoryFromIndex(index, data_directory))
        data_dir = patient_directory + b'/Original_Img_Data'
        for file in os.listdir(data_dir):
            img = self.getImage(data_dir+b'/'+file)
    ## Loads patient images and segments sequentially, assuming you want to through a range of numbered patients
    #
    # @param data_directory the directory where all patient data is located
    # @param start the patient number to start with (inclusive)
    # @param finish the patient number to stop at (exclusive)
    def load_images(self, data_directory, start, finish):
        print('Reading images')
        # Add classes
        self.add_class("mrimath", 1, "tumor")
        i = 0
        for j in range(start,finish):
            print('Reading Patient ' + str(j))
            patient_directory = os.fsencode(self.getDirectoryFromIndex(j, data_directory))
            data_dir = patient_directory + b'/Original_Img_Data'
            for file in os.listdir(data_dir):
                #img = self.getImage(data_dir+b'/'+file)
                seg_dir = patient_directory[len(patient_directory)-20:len(patient_directory)]
                seg_dir = b'/media/daniel/Backup Data/Ground_Truth' + seg_dir + file
                if np.sum(self.load_segment(seg_dir)) > 0:
                    self.add_image("mrimath", image_id=i, path=(data_dir+b'/'+file).decode('utf-8'), seg_dir=seg_dir)
                    i = i+1
    ## Loads patient images and segments in parallel, assuming you want to through a range of numbered patients
    #
    # @param data_directory the directory where all patient data is located
    # @param start the patient number to start with (inclusive)
    # @param finish the patient number to stop at (exclusive)
    def load_images_parallel(self, data_directory, start, finish):
        print('Reading images')
        self.X = self.manager.list()  
        self.labels = self.manager.list()  
        processes = []
        for i in range(start, finish):
            p = Process(target=self.loadIndividualPatient2, args=(i, data_directory))  
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

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
        info = self.image_info[image_id]
        seg_dir = info['seg_dir']
        mask = self.load_segment(seg_dir)
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
    
    ## Derives random patches from an image - Updated for February "pivot"
    #
    # @param patient_directoy the directory where the specific patient data is located (e.g. Patient_001_Data)
    # @param img the image to derive patches from
    # @param file the patient image number (e.g. img_1)
    def load_segment(self, seg_dir):
        if not os.path.exists(seg_dir):
            return
        mask_img = self.getImage(seg_dir)
        return mask_img



class InferenceConfig(MRIMathConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# stuff to run always here such as class/def
def main():
    config = MRIMathConfig()
    config.display()

    dataset_train = MRIMathDataset()
    dataset_train.load_images( '/media/daniel/Backup Data/Flair', 1,41)
    dataset_train.prepare()
    
        # Validation dataset
    dataset_val = MRIMathDataset()
    dataset_val.load_images( '/media/daniel/Backup Data/Flair', 41,62)
    dataset_val.prepare()
    
    
    #image_ids = np.random.choice(dataset_train.image_ids, 4)
    #for image_id in image_ids:
    #    image = dataset_train.load_image(image_id)
    #    mask, class_ids = dataset_train.load_mask(image_id)
    #    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
    
    
    # Load random image and mask.
    #image_id = random.choice(dataset_train.image_ids)
    #image = dataset_train.load_image(image_id)
    #mask, class_ids = dataset_train.load_mask(image_id)
    # Compute Bounding box
    #bbox = utils.extract_bboxes(mask)
    
    # Display image and additional stats
    #print("image_id ", image_id, dataset_train.image_reference(image_id))
    #log("image", image)
    #log("mask", mask)
    #log("class_ids", class_ids)
    #log("bbox", bbox)
    # Display image and instances
    #visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names)
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
                epochs=10, 
                layers='heads')


if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
    
