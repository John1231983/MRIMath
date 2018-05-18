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
import random
import os
import shutil

def computeDiceCoefficient(gt_boxes, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = utils.trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = utils.trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = utils.compute_overlaps_masks(pred_masks, gt_masks)
    return overlaps

inference_config = InferenceConfig()

test_dir = "Data/BRATS_2018/LGG_Testing"
data_dir = "Data/BRATS_2018/LGG"

dataset = MRIMathDataset()
dataset.load_images(test_dir)
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
jaccards[2] = []
jaccards[3] = []


djaccards = {}
djaccards[1] = []
djaccards[2] = []
djaccards[3] = []

precision = {}
precision[1] = []
precision[2] = []
precision[3] = []

recall = {}
recall[1] = []
recall[2] = []
recall[3] = []

dice = {}
dice[1] = []
dice[2] = []
dice[3] = []
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
            
    APs.append(AP)


print("Jaccard score for Core: " + str(np.mean(np.asarray(jaccards[1]))))
print("Jaccard score for Active: " + str(np.mean(np.asarray(jaccards[2]))))
print("Jaccard score for Whole: " + str(np.mean(np.asarray(jaccards[3]))))


print("Dice score for Core (from Jaccard): " + str(np.mean(np.asarray(djaccards[1]))))
print("Dice score for Active (from Jaccard): " + str(np.mean(np.asarray(djaccards[2]))))
print("Dice score for Whole (from Jaccard): " + str(np.mean(np.asarray(djaccards[3]))))

print("Precision for Core: " + str(np.mean(np.asarray(precision[1]))))
print("Precision for Active: " + str(np.mean(np.asarray(precision[2]))))
print("Precision for Whole: " + str(np.mean(np.asarray(precision[3]))))

print("Recall for Core: " + str(np.mean(np.asarray(recall[1]))))
print("Recall for Active: " + str(np.mean(np.asarray(recall[2]))))
print("Recall for Whole: " + str(np.mean(np.asarray(recall[3]))))

print("Dice for Core (from R+P): " + str(np.mean(np.asarray(dice[1]))))
print("Dice for Active (from R+P): " + str(np.mean(np.asarray(dice[2]))))
print("Dice for Whole (from R+P): " + str(np.mean(np.asarray(dice[3]))))

print("mAP: ", np.mean(AP))


# move the testing data back
list_imgs = os.listdir(test_dir)
for sub_dir in list_imgs:
    dir_to_move = os.path.join(test_dir, sub_dir)
    shutil.move(dir_to_move, data_dir)
        
        
        
        
        