# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 11:54:01 2018

@author: Fakrul-IslamTUSHAR
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pydicom.uid
import cv2
from glob import glob
import png


# =============================================================================
# Output paths
# =============================================================================
output_path_hist = 'D:/Preprocessint_Atlectesis_data_sep4/Atlectethesis/Histogram'
output_path_huunits ='D:/Preprocessint_Atlectesis_data_sep4/Atlectethesis/HU_UNITS'
Base_hu_img_path='D:/Preprocessint_Atlectesis_data_sep4/Atlectethesis/Base_hu_img'
NIH_low_hu_img_path='D:/Preprocessint_Atlectesis_data_sep4/Atlectethesis/NIH_low_hu_img'
NIH_normal_hu_img_path='D:/Preprocessint_Atlectesis_data_sep4/Atlectethesis/NIH_normal_hu_img'
NIH_high_hu_img_path='D:/Preprocessint_Atlectesis_data_sep4/Atlectethesis/NIH_high_hu_img'
Duke_low_hu_img_path='D:/Preprocessint_Atlectesis_data_sep4/Atlectethesis/Duke_low_hu_img'
Duke_high_hu_img_path='D:/Preprocessint_Atlectesis_data_sep4/Atlectethesis/Duke_high_hu_img'
Duke_Complete_hu_img_path='D:/Preprocessint_Atlectesis_data_sep4/Atlectethesis/Duke_Complete_hu_img'
# =============================================================================
# Functions
# =============================================================================
def get_pixels_hu(scans):
    image = np.stack([scans.pixel_array])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans.RescaleIntercept
    slope = scans.RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing =(float(scan.SliceThickness),float(scan.PixelSpacing[0]), float(scan.PixelSpacing[1]) )
    #spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

###Base Hu Units
MIN_BOUND = -1000.0
MAX_BOUND = 240.0    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

##NIH lOW Range
MIN_BOUND2 = -1400.0
MAX_BOUND2 = -950.0    
def normalize2(image):
    image = (image - MIN_BOUND2) / (MAX_BOUND2 - MIN_BOUND2)
    image[image>1] = 1.
    image[image<0] = 0.
    return image
###NIH Normal Range
MIN_BOUND3 = -1400.0
MAX_BOUND3 = 200.0   
def normalize3(image):
    image = (image - MIN_BOUND3) / (MAX_BOUND3 - MIN_BOUND3)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

###NIH High Range
MIN_BOUND4 = -160.0
MAX_BOUND4 = 240.0
def normalize4(image):
    image = (image - MIN_BOUND4) / (MAX_BOUND4 - MIN_BOUND4)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

###Duke Low Range
MIN_BOUND5 = -950.0
MAX_BOUND5 = -250.0
def normalize5(image):
    image = (image - MIN_BOUND5) / (MAX_BOUND5 - MIN_BOUND5)
    image[image>1] = 1.
    image[image<0] = 0.
    return image
###Duke High Range
MIN_BOUND6 = -250.0
MAX_BOUND6 = 250.0
def normalize6(image):
    image = (image - MIN_BOUND6) / (MAX_BOUND6 - MIN_BOUND6)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

###Duke Complete Range
MIN_BOUND7 = -1000.0
MAX_BOUND7 = 1000.0
def normalize7(image):
    image = (image - MIN_BOUND7) / (MAX_BOUND7 - MIN_BOUND7)
    image[image>1] = 1.
    image[image<0] = 0.
    return image



# =============================================================================
# Pre-Processing
# =============================================================================            
for im in glob('*.dcm'):
    ds = pydicom.read_file(im,force=True)
    filename_w_ext = os.path.basename(im)
    filename, file_extension = os.path.splitext(filename_w_ext)
    print filename          
# =============================================================================
# For logg
# =============================================================================

    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian  # or whatever is the correct transfer syntax for the file
    td=ds.pixel_array
    print(ds.pixel_array)

    ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), float(ds.SliceThickness))
# =============================================================================
#     getting the hu valuess
# =============================================================================
    #Hu_Value Getting
    first_patient_pixels = get_pixels_hu(ds)
    plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    #plt.show()
    
    plt.savefig(os.path.join(output_path_hist, filename+'_Histogram'+'.png'))
    cv2.imwrite(os.path.join(output_path_huunits, filename+'.png'),first_patient_pixels[0])
# Show some slice in the middle
    plt.imshow(first_patient_pixels[0], cmap=plt.cm.gray)
    #plt.show()
    
# =============================================================================
#     Resample
# =============================================================================
    pix_resampled, spacing = resample(first_patient_pixels, ds, [1,1,1])
    
# =============================================================================
#     Normalized
# =============================================================================
    
    ##Baase Hu window
    Base_hu_img=normalize(pix_resampled)
    ### NIH Low Hu window -1400 t0 -950
    NIH_low_hu_img=normalize2(pix_resampled)
    ### NIH Normal Hu window -1400 to 200
    NIH_normal_hu_img=normalize3(pix_resampled)
    #NIH High Hu window -160 to 240
    NIH_high_hu_img=normalize4(pix_resampled)
    #Duke low Hu range -950 to -250
    Duke_low_hu_img=normalize5(pix_resampled)
    # High Hu window -250 to 250
    Duke_high_hu_img=normalize6(pix_resampled)
    ###Duke Completw Hu units -1000 to 1000
    Duke_Complete_hu_img=normalize7(pix_resampled)


    #Rescaling grey scale between 0-255
    Base_hu_img1 = (np.maximum(Base_hu_img[0],0) / Base_hu_img[0].max()) * 255.0
    NIH_low_hu_img1 = (np.maximum(NIH_low_hu_img[0],0) / NIH_low_hu_img[0].max()) * 255.0
    NIH_normal_hu_img1 = (np.maximum(NIH_normal_hu_img[0],0) / NIH_normal_hu_img[0].max()) * 255.0
    NIH_high_hu_img1 = (np.maximum(NIH_high_hu_img[0],0) / NIH_high_hu_img[0].max()) * 255.0
    Duke_low_hu_img1 = (np.maximum(Duke_low_hu_img[0],0) / Duke_low_hu_img[0].max()) * 255.0
    Duke_high_hu_img1 = (np.maximum(Duke_high_hu_img[0],0) / Duke_high_hu_img[0].max()) * 255.0
    Duke_Complete_hu_img1 = (np.maximum(Duke_Complete_hu_img[0],0) / Duke_Complete_hu_img[0].max()) * 255.0

     # Convert to uint
    Base_hu_img12 = np.uint8(Base_hu_img1)
    NIH_low_hu_img12 = np.uint8(NIH_low_hu_img1)
    NIH_normal_hu_img12 = np.uint8(NIH_normal_hu_img1)
    NIH_high_hu_img12 = np.uint8(NIH_high_hu_img1)
    Duke_low_hu_img12  = np.uint8(Duke_low_hu_img1 )
    Duke_high_hu_img12= np.uint8(Duke_high_hu_img1)
    Duke_Complete_hu_img12= np.uint8(Duke_Complete_hu_img1)
    
   
    ####Saving the Images in Desired Folder
    cv2.imwrite(os.path.join(Base_hu_img_path, filename+'.png'),Base_hu_img12)
    cv2.imwrite(os.path.join(NIH_low_hu_img_path, filename+'.png'),NIH_low_hu_img12)
    cv2.imwrite(os.path.join(NIH_normal_hu_img_path, filename+'.png'),NIH_normal_hu_img12)
    cv2.imwrite(os.path.join(NIH_high_hu_img_path, filename+'.png'),NIH_high_hu_img12)
    cv2.imwrite(os.path.join(Duke_low_hu_img_path, filename+'.png'),Duke_low_hu_img12)
    cv2.imwrite(os.path.join(Duke_high_hu_img_path, filename+'.png'),Duke_high_hu_img12)
    cv2.imwrite(os.path.join( Duke_Complete_hu_img_path, filename+'.png'), Duke_Complete_hu_img12)