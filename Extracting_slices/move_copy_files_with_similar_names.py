"""
Created on Mon Aug 20 15:25:27 2018
@author: Fakrul-IslamTUSHAR
"""

# =============================================================================
"""
Details: ***This Script to Move same name files from the directory
"""
# =============================================================================

# =============================================================================
# Import libraries
# =============================================================================
import shutil
import os
import pandas

###This is the Directory Where you have the images you want to get the name and compare with the image Directory
Source_dir1=os.path.join('C:/Users/Fakrul-IslamTUSHAR/Desktop/chv1/train_1/edema')
Source_dir1_list = os.listdir(Source_dir1)

####This the directory from where you want to move the same_name Images to Destination Directory
img_dir=os.path.join('D:/Preprocessint_Atlectesis_data_AUG27/edema/Duke_low_hu_img')
img_list_dir = os.listdir(img_dir)

###This is the Destination directory
destination= os.path.join('C:/Users/Fakrul-IslamTUSHAR/Desktop/AUG_28/edema/ChannelV3_Duke/Train_3_Duke_low_hu/edema')


##For-loop for move the files
for sub_dir in Source_dir1_list: #This will run a for-loop to the length of the Sorce_Directory_1
    if sub_dir in img_list_dir:  #tHis comoares the name of the images in both directory
        dir_to_move = os.path.join(img_dir, sub_dir) #which one to move
        shutil.copy(dir_to_move, destination) #where to move= move, copy=where to copy


        

