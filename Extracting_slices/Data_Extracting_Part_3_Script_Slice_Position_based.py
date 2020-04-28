# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 23:29:21 2018

@author: Fakrul-IslamTUSHAR
"""

"""
his Script for Getting the Normal and Abnormal data folders from the main dataset.

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
    3.Then From the Normal And Abnormal Folder we need to select 10 Slices from each patient depending on the slice position of the dicom file>>>>>>Data_Extracting_Part_3_Script_slice_position_besed
        process:
        3.1>>go through all the patients dicom folder of the multiple screening.
        3.2>>Read all the dicom file and get the pistion of the slice,
        3.3>> If the position of the slice is between abs(-110) to (-225) then it will select the slice.
        3.4>>Then store the selected slice to the destination folder
   alternative of 3 is using slice index also used in >Data_Extracting_Part_3_Script_slice_index_based

"""


# =============================================================================
# imPORT Libraries
# =============================================================================
import numpy as np
import shutil
import dicom # for reading dicom files
import os # for doing directory operations 
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)

# Change this to wherever you are storing your data:
# IF YOU ARE FOLLOWING ON KAGGLE, YOU CAN ONLY PLAY WITH THE SAMPLE DATA, WHICH IS MUCH SMALLER

data_dir = 'D:/Althesis_Agu14'
destination='D:/Althesis_Slices_Aug27'
patients = os.listdir(data_dir)
#len(patients)
for patient in range(200,len(patients)):
    path = data_dir +'/'+ patients[patient]
    print(path)
    Scanning_list=os.listdir(path) #taking the list of the images in each folder
    for scanning in range(0,len(Scanning_list)):
        recent_directory= os.path.join(path+'/'+Scanning_list[scanning])
        print(recent_directory)
    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
        Dicom_slice_list=os.listdir(recent_directory)
        slices = [dicom.read_file(recent_directory + '/' + s) for s in os.listdir(recent_directory)]
        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
        
        location_list=np.arange(110, 255, 10)
       
        
        for number_of_slice in range(0,len(Dicom_slice_list)):
            if (round(abs(slices[number_of_slice].SliceLocation)) in location_list) :
               #print(len(slices))
               #print(round(abs(slices[number_of_slice].SliceLocation)))
               print(Dicom_slice_list[number_of_slice])
               print(slices[number_of_slice].SliceLocation)
               dir_to_move2 = os.path.join(recent_directory, Dicom_slice_list[number_of_slice])
               shutil.copy(dir_to_move2, destination)
               
