# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 03:06:10 2018

@author: Fakrul-IslamTUSHAR

This Script for Getting the Normal and Abnormal data folders from the main dataset.

Data Data details:
    Data that we are dealing with , as 3 step.
    1.there is folder which has folder with patient dicom folder in it.
    2.In side the patient folder there are multiple Screening data
    3.In side that data Folder

Data Extracting:
    There are three part of getting the slices.
    1. we go throuw all the folder and get the patients folder of the main Raw data folder using this >>Data_Extracting_Part_1_Script
    2. Thn from the extracted patients folder we need to seperate the Normal and abnormal 
    patients comparing from a excel sheet has the all patient number and state using>>>Data_Extracting_Part_2 Script
    3.Then From the Normal And Abnormal Folder we need to select 10 Slices from each patient depending on the slice position of the dicom file>>>>>>Data_Extracting_Part_3 Script
        process:
        3.1>>go through all the patients dicom folder of the multiple screening.
        3.2>>Read all the dicom file and get the pistion of the slice,
        3.3>> If the position of the slice is between abs(-110) to (-225) then it will select the slice.
        3.4>>Then store the selected slice to the destination folder
    3d.I also do one based on the slice index.

"""

# =============================================================================
# Import Libraries
# =============================================================================
import shutil
import os
import pandas

###Source Folder
cur_dir=os.path.join('D:/Deidentify_Batch2_Aug5')
list_dir = os.listdir(cur_dir)

##Destination Folder
dest = os.path.join('D:/Althesis') 

##go through alll the folders and get the Patient Folders
for subfolder in list_dir:
   
    img_list=os.listdir(cur_dir+'/'+subfolder) #taking the list of the images in each folder
    print('Loaded the image of dataset-'+ '{}\n'.format(subfolder))
    print('Loaded the image of dataset-'+ '{}\n'.format(img_list))
    print(img_list)
    for i in range(0,len(img_list)):
       dir_to_move = os.path.join(cur_dir+'/'+str(subfolder)+'/'+img_list[i])
       shutil.move(dir_to_move, dest)