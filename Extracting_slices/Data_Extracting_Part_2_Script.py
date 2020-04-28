# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 01:31:34 2018
###Move Folder
@author: Fakrul-IslamTUSHAR


This code is to identify the Normal And Abnormal Cases and put them in seperate folder
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
    patients comparing from a excel sheet has the all patient number and state using>>>Data_Extracting_Part_2 Script
    3.Then From the Normal And Abnormal Folder we need to select 10 Slices from each patient depending on the slice position of the dicom file>>>>>>Data_Extracting_Part_3 Script
        process:
        3.1>>go through all the patients dicom folder of the multiple screening.
        3.2>>Read all the dicom file and get the pistion of the slice,
        3.3>> If the position of the slice is between abs(-110) to (-225) then it will select the slice.
        3.4>>Then store the selected slice to the destination folder

"""

# =============================================================================
# Import Libraries
# =============================================================================
import shutil
import os
import pandas
# =============================================================================
# Load Excel File
# =============================================================================
Read_Normal_Case_csv= pandas.read_csv('normal.csv',header=None) #LOad CSV file of labels
x =Read_Normal_Case_csv.to_string(header=False,index=False,index_names=False).split('\n') #Making the list
Normal_Case_List= [','.join(ele.split()) for ele in x] #plit by ,
print(Normal_Case_List)


# =============================================================================
# Folder Move
# =============================================================================
#cur_dir = os.getcwd() # current dir path
cur_dir=os.path.join('D:/Althesis')
#Listing the Folder In the directory.
list_dir = os.listdir(cur_dir)
#Destination Directoey 
dest = os.path.join('D:/Normal_Aug14') 

for sub_dir in list_dir:
    if sub_dir in Normal_Case_List:
        dir_to_move = os.path.join(cur_dir, sub_dir)
        shutil.move(dir_to_move, dest)