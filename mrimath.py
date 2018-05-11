import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib


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
    NUM_CLASSES = 1 + 2  # background + 2 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16,32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 200
    BACKBONE = "resnet101"


    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
    
class MRIMathDataset(utils.Dataset):
            
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        img = np.zeros(shape=(240,240,3))
        j = 0
        info = self.image_info[image_id]
        for path in os.listdir(info['path']):
            if "flair" in path or "t2" in path or "t1ce" in path:
                img[:,:,j] = nib.load(self.image_info[image_id]['path'] + "/"+path).get_data()[:,:,info['ind']]
                j = j+1
        return img

    ## Loads patient images and segments sequentially, assuming you want to through a range of numbered patients
    #
    # @param data_directory the directory where all patient data is located
    # @param start the patient number to start with (inclusive)
    # @param finish the patient number to stop at (exclusive)
    def load_images(self, data_dir):
        print('Reading images')
        # Add classes
        self.add_class("mrimath", 1, "core")
        self.add_class("mrimath", 2, "active")
        #self.add_class("mrimath", 3, "whole")

        i = 0
        for subdir in os.listdir(data_dir):
            for j in range(0,155):
                self.add_image("mrimath", image_id=i, path=data_dir + "/" + subdir, ind=j)
                i = i+1

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

        mask = np.zeros(shape = (240,240,2))
        for path in os.listdir(self.image_info[image_id]['path']):
            if "seg" in path:
                seg = nib.load(self.image_info[image_id]['path']+"/"+path).get_data()[:,:,self.image_info[image_id]['ind']]
                break
        
        class_ids = []
        seg_active = seg.copy()
        seg_active[seg > 3] = 0
        seg_active[seg<3] = 2
        seg_active[seg==0] = 0
        
        seg_core = seg.copy()

        seg_core[seg > 1]=0
        if seg_core.any():
            class_ids.append(1)
        else:
            class_ids.append(0)

        
        if seg_active.any():
            class_ids.append(2)
        else:
            class_ids.append(0)
        
        
        """
        seg_whole = seg.copy()
        if 4 in seg_whole[:,:]:
            seg_whole[seg > 0] = 4
            class_ids.append(3)
        else:
            seg_whole = np.zeros((seg_whole.shape[0], seg_whole.shape[1]))
            class_ids.append(0)
        """
        
        """
        plt.figure(1)
        plt.subplot(311)
        plt.imshow(seg)
        plt.subplot(312)
        plt.imshow(seg_core)
        plt.subplot(313)
        plt.imshow(seg_active)

        plt.show()
        """
        
    
        mask[:,:,0] = seg_core
        mask[:,:,1] = seg_active
        #mask[:,:,2] = seg_whole

        
        return mask.astype(np.bool),np.array(class_ids).astype(np.int32)
    
    ## Derives random patches from an image - Updated for February "pivot"
    #
    # @param patient_directoy the directory where the specific patient data is located (e.g. Patient_001_Data)
    # @param img the image to derive patches from
    # @param file the patient image number (e.g. img_1)
    def load_segment(self, seg_dir):
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
    dataset_train.load_images("Data/BRATS_2018/LGG")
    dataset_train.prepare()
    
    
    dataset_val = MRIMathDataset()
    dataset_val.load_images("Data/BRATS_2018/LGG_Validation")
    dataset_val.prepare()
    
    print(len(dataset_train.image_info))
        # Validation dataset
    #dataset_val = MRIMathDataset()
    #dataset_val.load_images( '/media/daniel/Backup Data/Flair', 130,180)
    #dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    # Which weights to start with?
    init_with = "last"  # imagenet, coco, or last
    
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
                epochs=300,
                layers='all')


if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
    
