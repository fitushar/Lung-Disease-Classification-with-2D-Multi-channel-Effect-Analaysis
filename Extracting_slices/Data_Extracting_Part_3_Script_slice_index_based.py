# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 12:37:42 2018

@author: Fakrul-IslamTUSHAR
"""
"""
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
    patients comparing from a excel sheet has the all patient number and state using>>>Data_Extracting_Part_3 Script
    3.Then From the Normal And Abnormal Folder we need to select 10 Slices from each patient depending on the slice position of the dicom file>>>>>>Data_Extracting_Part_3 Script_slice_position_based
        process:
        3.1>>go through all the patients dicom folder of the multiple screening.
        3.2>>Read all the dicom file and get the pistion of the slice,
        3.3>> If the position of the slice is between abs(-110) to (-225) then it will select the slice.
        3.4>>Then store the selected slice to the destination folder
        
     alternativ: Data_Extracting_Part_3 Script_slice_index_based

"""

# =============================================================================
# IMport Libraries
# =============================================================================

import shutil
import os
import pandas

#Sorce Folder
cur_dir=os.path.join('D:/Normal_Aug14')
list_dir = os.listdir(cur_dir)

 #making list to store all the folder
###Destination Folder
dest = os.path.join('D:/Normal_Slice') 

for subfolder in list_dir:
   
    img_list=os.listdir(cur_dir+'/'+subfolder) #taking the list of the images in each folder
    print('Loaded the image of dataset-'+ '{}\n'.format(subfolder))
    print('Loaded the image of dataset-'+ '{}\n'.format(img_list))
    print(img_list)
    #print('Folder_Name:',img_list)
    #df=img_list.split('\n')
    #df=str(img_list).strip('[]')
    #print('value:',df)
    for i in range(0,len(img_list)):
       print(img_list[i])
       recent_directory= os.path.join(cur_dir+'/'+str(subfolder)+'/'+img_list[i])
       print(recent_directory)
       list_dir2 = os.listdir(recent_directory)
       print(list_dir2)
#       shutil.move(dir_to_move, dest)
       
       con_val=len(list_dir2)/5
       
       if len(list_dir2)>200:
          slice_list=[list_dir2[100],list_dir2[110],list_dir2[120],list_dir2[130],list_dir2[140],list_dir2[150],list_dir2[160],list_dir2[170],list_dir2[180],list_dir2[190]]
       else:
          slice_list=[list_dir2[con_val],list_dir2[con_val+10],list_dir2[con_val+20],list_dir2[con_val+30],list_dir2[con_val+40]]
        
       for sub_dir2 in list_dir2:
        if sub_dir2 in slice_list:
         dir_to_move2 = os.path.join(recent_directory, sub_dir2)
         shutil.copy(dir_to_move2, dest)
        