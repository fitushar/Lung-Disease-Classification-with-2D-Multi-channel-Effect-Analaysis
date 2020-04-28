"""
Created on Wed Aug 26 23:29:21 2018

@author: Fakrul-IslamTUSHAR


This Script for Getting get 20-25 slices from the patients to make a average histogram.
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
# =============================================================================
# fEW LIST
# =============================================================================
slice_thikness_list=[]
Number_of_slices_list=[]
Number_of_slices_moved_list=[]
data_dir = 'D:/histoGram_Related_Issue/sulu_folder/sulu2/new_Histogram Check/edema'
destination='D:/histoGram_Related_Issue/sulu_folder/sulu2/new_Histogram Check/destination'
patients = os.listdir(data_dir)
#len(patients)
for patient in range(0,len(patients)):
    path = data_dir +'/'+ patients[patient]
    print(path)
    Scanning_list=os.listdir(path) #taking the list of the images in each folder
    for scanning in range(0,len(Scanning_list)):
        recent_directory= os.path.join(path+'/'+Scanning_list[scanning])
        print(recent_directory)
    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
        Dicom_slice_list=os.listdir(recent_directory)
        print("Total Number of slices={}".format(len(Dicom_slice_list)))
        Number_of_slices_list.append(len(Dicom_slice_list))
        slices = dicom.read_file(recent_directory + '/' +Dicom_slice_list[0])
        print("Slice_Thinkness={}".format(slices.SliceThickness))
        slice_thikness_list.append(slices.SliceThickness)
        #slices = [dicom.read_file(recent_directory + '/' + s) for s in os.listdir(recent_directory)]
#        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
        array=np.arange(0,len(Dicom_slice_list),abs(len(Dicom_slice_list)/20))
        ###Defining a list for making the Slice List
        dicom_slice_index=[]
        
        for i in range(0,len(array)):
           dicom_slice_index.append(Dicom_slice_list[array[i]])

        
        print('Number of slices moved={}'.format(len(dicom_slice_index)))
        Number_of_slices_moved_list.append(len(dicom_slice_index))
        print(dicom_slice_index)
        for number_of_slice in Dicom_slice_list:
           
            #for number_of_slice in list_dir2:
            if not number_of_slice in dicom_slice_index:
               dir_to_move2 = os.path.join(recent_directory, number_of_slice)
               shutil.move(dir_to_move2, destination)
               

               
