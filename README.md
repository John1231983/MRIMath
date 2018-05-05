# MRIMath

This is an implementation of an brain MRI segmentation system, MRIMATH. There are two implementations of the system thus far -
one which is done using a traditional CNN and decomposes segmentation as a classification problem to derive NMF-LSM segments, and the other which leverages 
a Mask R-CNN implementation. 

Datahandler.py, LoadAndTestModel.py, and TrainModels.py are all portions of the first implementation.
mrimath.py and Visualizations.py are part of the second implementation (which also depend on https://github.com/matterport/Mask_RCNN)

HardwareHandler.py and EmailHandler.py are general tools which assist with multithreading, multi-GPU use, and sending emails when processes finish.

Note: this is currently a WIP - as work becomes more concrete, this will become more comprehensive and detailed. For the time being, if you have any questions about things which are not clear, feel free to shoot me an email at danielenricocahall@gmail.com
