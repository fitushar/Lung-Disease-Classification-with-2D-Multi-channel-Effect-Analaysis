# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 22:28:19 2018
This Script is for making the slice histogram.
This Sript take 20 slices from each patient and gives a patientwise and a average of all patient histogram

@author: Fakrul-IslamTUSHAR

"""
# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import dicom
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

##Making a list to store the All patient Data
One_list_for_all_histogram_values=[]
# =============================================================================
# Functions
# =============================================================================
###Loading the slices in the disting folder
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

###Getting THE HU units
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

# =============================================================================
# 
# =============================================================================
###Provide the path of the Folder that contains the images.
data_path='D:/histoGram_Related_Issue/sulu_folder/sulu2/new_Histogram Check/hist_40_data'
List_of_patients=os.listdir(data_path)

##This will take loop through the all patient
for patient in range(0,len(List_of_patients)):
    patient_path=os.path.join(data_path+'/'+List_of_patients[patient])
    print(patient_path)
    
    ###Get the Slice list in a patient
    Number_of_slices=os.listdir(patient_path)
    load_patient_slices = load_scan(patient_path)
    imgs = get_pixels_hu(load_patient_slices)
    flaten_all_slice_value=imgs.flatten()
    
    ##Give the histogram of a patient
    output_path_slice_voxel_values='D:/histoGram_Related_Issue/sulu_folder/sulu2/new_Histogram Check/save_voxel_values'
    save_data_name=List_of_patients[patient]+'.npy'
    np.save(os.path.join(output_path_slice_voxel_values,save_data_name), imgs)
    
    One_list_for_all_histogram_values.append(flaten_all_slice_value)
    
    plt.hist(flaten_all_slice_value, bins=50, color='c')
    plt.title('Histogra of patient={},total Number of Slices= {}'.format(List_of_patients[patient],len(Number_of_slices)))
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    
    ###Saving patient Histogram
    Histogram_name='Histogra of patient={},total Number of Slices= {}'.format(List_of_patients[patient],len(Number_of_slices))+'.png'
    output_path_hist = 'D:/histoGram_Related_Issue/sulu_folder/sulu2/new_Histogram Check/Patient_Histogram'
    plt.savefig(os.path.join(output_path_hist, Histogram_name))
    plt.show()
    
    
    ###Saving the 
    plt.title('Average Histogra-patient#{},Slices#{} voxels#{}'.format(List_of_patients[patient],len(Number_of_slices),len(flaten_all_slice_value)))
    weights = np.ones_like(flaten_all_slice_value)/float(len(flaten_all_slice_value))
    plt.hist( flaten_all_slice_value, weights=weights,bins=50, color='c')
    Avg_hist_name='Average Histogra-patient#{},Slices#{} voxels#{}'.format(List_of_patients[patient],len(Number_of_slices),len(flaten_all_slice_value)) +'png'    
    output_path_Avg_hist = 'D:/histoGram_Related_Issue/sulu_folder/sulu2/new_Histogram Check/Average_patient_histogram'
    plt.savefig(os.path.join(output_path_Avg_hist, Avg_hist_name))
    plt.show()
    


####Taking the All the voxels values for all the patients
tusahar=np.concatenate( One_list_for_all_histogram_values, axis=0 )


####MAKING THW 50 PATIENT HIST
print(tusahar)
plt.hist(tusahar, bins=50, color='c')
plt.title('40 Patients Hist,#Voxel= {}'.format(len(tusahar)))
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
Histogram_name_50='40 Patients Hist,#Voxel= {}'.format(len(tusahar))+'.png'
output_path_hist2 = 'D:/histoGram_Related_Issue/sulu_folder/sulu2/new_Histogram Check'
plt.savefig(os.path.join(output_path_hist2, Histogram_name_50))
plt.show()
#
######Normalized
plt.title('Average Histogram 40 Patients,voxels#{}'.format(len(tusahar)))
weights2 = np.ones_like(tusahar)/float(len(tusahar))
plt.hist( tusahar, weights=weights2,bins=50, color='c')
Avg_hist_name2='Average Histogram 40 Patients,voxels#{}'.format(len(tusahar)) +'png'    
#plt.savefig(os.path.join(output_path_hist2, Avg_hist_name2))
plt.show()




# =============================================================================
# This Section is for customized ploting you can manupulate the x and y axis limit by this part
# =============================================================================
output_path_histdude='C:/Users/Fakrul-IslamTUSHAR/Desktop/la_images'
plt.title('HU Range[-950,-250],voxels#{}'.format(len(tusahar)))
weights2 = np.ones_like(tusahar)/float(len(tusahar))
plt.hist( tusahar, weights=weights2,bins=80, color='c')
Avg_hist_name3='NIH_High_Range,voxels#{}'.format(len(tusahar)) +'png'
plt.ylabel('% of Voxels')
plt.xlabel('HU units')    
#plt.savefig(os.path.join(output_path_histdude,Avg_hist_name3))
plt.xticks(range(-950, -250),color='c')
plt.xlim(xmin=-1400, xmax =1000)
#plt.ylim(ymin=0, ymax =10 )
plt.show()




    
    
    
    
    
