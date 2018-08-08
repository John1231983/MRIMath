'''
Created on Aug 1, 2018

@author: daniel
'''
from Exploratory_Stuff.DataHandler import DataHandler
import numpy as np
import cv2
class SegNetDataHandler(DataHandler):
    
    def __init__(self,dataDirectory, nmfComp, W = 240, H = 240, num_patients = 3):
        super().__init__(dataDirectory, nmfComp, W, H, num_patients)
        
        
    def performNMFOnSlice(self, image, seg_image, i):
        # image[:,:,i] = self.preprocess(image[:,:,i])        
        W, H = self.nmfComp.run(image[:,:,i])
        return self.processData(image[:,:,i], W,H, seg_image[:,:,i])
    
    def processData(self, image, W, H, seg_image):
        X = []
        y = []
        indices = np.argmax(W, axis=0)
        #H = H[indices > 0]
        regions = np.argmax(H, axis=0)
        
        
        
        region_1 = regions.copy()
        region_1_and_2 = regions.copy()
        region_1_and_2_and_3 = regions.copy()
        region_1_and_2_and_3_and_4 = regions.copy()
        region_1_and_2_and_3_and_4_and_5 = regions.copy()
        region_1_and_2_and_3_and_4_and_5_and_6 = regions.copy() 
        
        region_1[regions < 0] = 0
        region_1[regions > 1] = 0
        region_1 = region_1.astype(bool)
        
        region_1_and_2[regions < 1] = 0
        region_1_and_2[regions > 2] = 0
        region_1_and_2 = region_1_and_2.astype(bool)
                
        region_1_and_2_and_3[regions < 1] = 0
        region_1_and_2_and_3[regions > 3] = 0
        region_1_and_2_and_3 = region_1_and_2_and_3.astype(bool)
        
        region_1_and_2_and_3_and_4[regions < 1] = 0
        region_1_and_2_and_3_and_4[regions > 4] = 0
        region_1_and_2_and_3_and_4 = region_1_and_2_and_3_and_4.astype(bool)
        
        region_1_and_2_and_3_and_4_and_5[regions < 1] = 0
        region_1_and_2_and_3_and_4_and_5[regions > 5] = 0
        region_1_and_2_and_3_and_4_and_5 = region_1_and_2_and_3_and_4_and_5.astype(bool)

        
        region_1_and_2_and_3_and_4_and_5_and_6[regions < 1] = 0
        region_1_and_2_and_3_and_4_and_5_and_6[regions > 6] = 0
        region_1_and_2_and_3_and_4_and_5_and_6 = region_1_and_2_and_3_and_4_and_5_and_6.astype(bool)
        
        reg_1_image = image.copy()
        reg_1_and_2_image = image.copy()
        reg_1_and_2_and_3_image = image.copy()
        reg_1_and_2_and_3_and_4_image = image.copy()
        reg_1_and_2_and_3_and_4_and_5_image = image.copy()
        reg_1_and_2_and_3_and_4_and_5_and_6_image = image.copy()
        
        #regions = regions.astype(bool)
        m = self.nmfComp.block_dim
        ind = 0
        for i in range(0, seg_image.shape[0], m):
            for j in range(0, seg_image.shape[1], m):
                reg_1_image[i:i+m, j:j+m] *= region_1[ind]
                reg_1_and_2_image[i:i+m, j:j+m] *= region_1_and_2[ind]
                reg_1_and_2_and_3_image[i:i+m, j:j+m] *= region_1_and_2_and_3[ind]
                reg_1_and_2_and_3_and_4_image[i:i+m, j:j+m] *= region_1_and_2_and_3_and_4[ind]
                reg_1_and_2_and_3_and_4_and_5_image[i:i+m, j:j+m] *= region_1_and_2_and_3_and_4_and_5[ind] 
                reg_1_and_2_and_3_and_4_and_5_and_6_image[i:i+m, j:j+m] *= region_1_and_2_and_3_and_4_and_5_and_6[ind]
                ind = ind + 1
                """
        seg_image[seg_image > 0] = 1
        """
        
        dims = tuple([self.W, self.W])
        seg_image[seg_image > 0] = 1
        #seg_image = self.label_map(seg_image, 2)
        
        image = cv2.resize(image,dims)
        X.append(image)
        y.append(seg_image)
        
        reg_1_and_2_and_3_image = cv2.resize(reg_1_and_2_and_3_image,dims)
        X.append(reg_1_and_2_and_3_image)
        y.append(seg_image)
        
        reg_1_and_2_and_3_and_4_image = cv2.resize(reg_1_and_2_and_3_and_4_image,dims)
        X.append(reg_1_and_2_and_3_and_4_image)
        y.append(seg_image)
        
        reg_1_and_2_and_3_and_4_and_5_image = cv2.resize(reg_1_and_2_and_3_and_4_and_5_image,dims)
        X.append(reg_1_and_2_and_3_and_4_and_5_image)
        y.append(seg_image)

        reg_1_and_2_and_3_and_4_and_5_and_6_image = cv2.resize(reg_1_and_2_and_3_and_4_and_5_and_6_image,dims)
        X.append(reg_1_and_2_and_3_and_4_and_5_and_6_image)
        y.append(seg_image)
        
        
        
        """
        fig = plt.figure()
        plt.gray();   
        
        a=fig.add_subplot(1,8,1)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Original')
        
        a=fig.add_subplot(1,8,2)
        plt.imshow(reg_1_image)
        plt.axis('off')
        plt.title('1')
        
        
        a=fig.add_subplot(1,8,3)
        plt.imshow(reg_1_and_2_image)
        plt.axis('off')
        plt.title(r'1 $\cup$ 2')
        
        a=fig.add_subplot(1,8,4)
        plt.imshow(reg_1_and_2_and_3_image)
        plt.axis('off')
        plt.title(r'1 $\cup$ 2 $\cup$ 3')
        self.X.append(reg_1_and_2_and_3_image)
        
        
        a=fig.add_subplot(1,8,5)
        plt.imshow(reg_1_and_2_and_3_and_4_image)
        plt.axis('off')
        plt.title(r'1 $\cup$ 2 $\cup$ 3 $\cup$ 4')
        
        a=fig.add_subplot(1,8,6)
        plt.imshow(reg_1_and_2_and_3_and_4_and_5_image)
        plt.axis('off')
        plt.title(r'1 $\cup$ 2 $\cup$ 3 $\cup$ 4 $\cup$ 5')


        a=fig.add_subplot(1,8,7)
        plt.imshow(reg_1_and_2_and_3_and_4_and_5_and_6_image)
        plt.axis('off') 
        plt.title(r'1 $\cup$ 2 $\cup$ 3 $\cup$ 4 $\cup$ 5 $\cup$ 6')

        a=fig.add_subplot(1,8,8)
        plt.imshow(seg_image)
        plt.axis('off')
        plt.title('Segment')

        plt.show()
        """
        return X, y
    
    def label_map(self, labels, n_labels):
        label_map = np.zeros([self.W, self.H, n_labels])    
        for r in range(self.W):
            for c in range(self.W):
                label_map[r, c, labels[r][c]] = 1
        label_map = label_map.reshape(self.W * self.H, n_labels)
        return label_map
    
    
    def preprocessForNetwork(self):
        n_imgs = len(self.X)
        self.X = np.array( self.X )
        self.X = self.X.reshape(n_imgs,self.W, self.H,1)
        self.labels = np.array( self.labels )
        #self.labels = self.labels.reshape(n_imgs,self.W*self.H,2)
        # self.labels = self.labels.reshape(n_imgs, self.W*self.H,2)