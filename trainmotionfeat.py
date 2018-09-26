"""
  This script is used to train the motion features of the video, the work is just started
  so you would not able to use it now...
  
"""
import numpy as np
import cv2
import os, os.path

# Here chanage the path to the train and dataset folder

datasetPath = "UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
samplePath = datasetPath = "/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001"
dPath = os.listdir(datasetPath)	# List of all the folder in the train folder
dPath = sorted(dPath)[2:] # Sort and remove the ._.DS_Store, .DS_Strore folder

print (datasetPath)
print (dPath)
subfolders = []
# These are the samples folders which has sliced images for different features
# optframe - Sliced images for optical frame in color
# optgrayframe - Sliced images for optical frame in grayscale
# wincolor - Sliced images of original video color
# wingray - Sliced images of originial video in gray scale
sampleFolders = ['optframe', 'optgrayframe', 'wincolor', 'wingray']

# for 
