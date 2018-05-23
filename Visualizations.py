import sys
import os
ROOT_DIR = os.path.abspath("Mask_RCNN")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrimath import InferenceConfig, FlairDataset
import numpy as np
import random
import os
import shutil

inference_config = InferenceConfig()

test_dir = "Data/BRATS_2018/HGG_Testing"
data_dir = "Data/BRATS_2018/HGG"

dataset = FlairDataset()
dataset.load_images(test_dir)
dataset.prepare()
print("Testing on " + str(len(dataset.image_info)) + " images")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = model.find_last()[1]
# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True) 

# Test on a random image

image_id = random.choice(dataset.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, inference_config, 
                           image_id, use_mini_mask=False)
print(dataset.image_info[image_id])
log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

results = model.detect([original_image], verbose=0)
r = results[0]

#visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
 #                           dataset.class_names, figsize=(8, 8))
visualize.display_differences(original_image,
                        gt_bbox, gt_class_id, gt_mask,
                        r["rois"], r["class_ids"], r["scores"], r['masks'],
                        dataset.class_names)     

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
#image_ids = np.random.choice(dataset.image_ids, 1000)
#print(image_ids)
#print(dataset.image_ids)
APs = []


jaccards = {}
jaccards[1] = []


djaccards = {}
djaccards[1] = []

precision = {}
precision[1] = []

recall = {}
recall[1] = []

dice = {}
dice[1] = []
for image_id in dataset.image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    
    # Compute AP 
    if ((r['masks'] > 0.5).size <= 0 or (gt_mask > 0.5).size <= 0): 
        continue
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                        r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=0.0)
        
    for i,label in enumerate(r["class_ids"]):
        if label in gt_class_id:
            jaccards[label].append(overlaps[i])
            djaccards[label].append(2*overlaps[i]/(1+overlaps[i]))
            precision[label].append(precisions[i])
            recall[label].append(recalls[i])
            harmonic_mean = 2*recalls[i]*precisions[i]
            if harmonic_mean > 0:
                harmonic_mean = harmonic_mean/(precisions[i] + recalls[i])
            dice[label].append(harmonic_mean)
            

            #print(jaccards[label])
    #print(AP)      
    APs.append(AP)


print("Jaccard score for Core: " + str(np.mean(np.asarray(jaccards[1]))))

print("Dice score for Core (from Jaccard): " + str(np.mean(np.asarray(djaccards[1]))))

print("Precision for Core: " + str(np.mean(np.asarray(precision[1]))))

print("Recall for Core: " + str(np.mean(np.asarray(recall[1]))))

print("Dice for Core (from R+P): " + str(np.mean(np.asarray(dice[1]))))

print("mAP: ", np.mean(APs))


# move the testing data back
"""
list_imgs = os.listdir(test_dir)
for sub_dir in list_imgs:
    dir_to_move = os.path.join(test_dir, sub_dir)
    shutil.move(dir_to_move, data_dir)
"""
        
        
        
        
        