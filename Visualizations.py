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
from mrimath import InferenceConfig, MRIMathDataset
import numpy as np

inference_config = InferenceConfig()

dataset = MRIMathDataset()
dataset.load_images("Data/BRATS_2018/LGG_Testing")
dataset.prepare()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
#model_path = os.path.join(ROOT_DIR, "mask_rcnn_mrimath_0010.h5")
model_path = model.find_last()[1]
# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True) 
"""
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
"""
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
#image_ids = np.random.choice(dataset.image_ids, 1200)
#print(image_ids)
#print(dataset.image_ids)
APs = []
dices = []
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
                        r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=0.5)
    APs.append(AP)
    """
    dice = 0
    for i in range(0, r['masks'].shape[2]):
        temp = r['masks'][:,:,i]
        intersection = np.logical_and(temp, gt_mask[:,:,i])
        seg_sum = r['masks'][:,:,i].sum() + gt_mask[:,:,i].sum()
        if seg_sum > 0:
            dice = dice + 2.*intersection.sum()/seg_sum
    dices.append(dice)
    """
    

    
print("Dice: ", np.mean(APs))