# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 13:27:29 2018
TRying to make three channel with different image
@author: Fakrul-IslamTUSHAR
"""
import os
import cv2
import numpy as np
# =============================================================================
# Load Image
# =============================================================================
##Loading the data
PATH=os.getcwd()
##Giving the path of the directory
data_path1=PATH+'/train_1'
data_path2=PATH+'/train_2'
data_path3=PATH+'/train_3'
##
data_dir_list=os.listdir(data_path1)##getting the list of the folder in the directory.
print(data_dir_list)
#
img_data_list=[] #making list to store all the images
#
for dataset in data_dir_list:
    img_list=os.listdir(data_path1+'/'+dataset) #taking the list of the images in each folder
    print('Loaded the image of dataset-'+ '{}\n'.format(dataset))
    
    for img in img_list:
        
# =============================================================================
#LOAD Image for Channel One
# =============================================================================
        img_path1=data_path1+'/'+dataset+'/'+img #getting the image path
        img1 = cv2.imread((img_path1),-1)
        img1 = cv2.resize(img1,(224,224))
# =============================================================================
# LOAD Image for Channel two        
# =============================================================================
        img_path2=data_path2+'/'+dataset+'/'+img
        img2 = cv2.imread((img_path2),-1)
        img2 = cv2.resize(img2,(224,224))
# =============================================================================
#LOAD Image for Channel three         
# =============================================================================
        img_path3=data_path2+'/'+dataset+'/'+img
        img3 = cv2.imread((img_path3),-1)
        img3 = cv2.resize(img3,(224,224))
# =============================================================================
# you can use ur one channel to make three channel        
# =============================================================================
        img4 = np.tile(img1,(1,1,1))
# =============================================================================
#You can use all the three one channel Image to get three channel Image
# =============================================================================
        img5 = np.dstack((img1,img2,img3))
#        #print('input image shape:',img2.shape)
        cv2.imwrite("original2.png",img5)
#        #x=image.img_to_array(img2) #converting the image to array
#        #x=np.expand_dims(x,axis=0) #putting them in the row axis
#        #x=preprocess_input(x) #preprocessing
#        print('input image shape:',img2.shape)
#        #cv2.imwrite("Final11.png",img2)
#        img_data_list.append(img2)
#
