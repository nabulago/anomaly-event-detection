import numpy as np
import cv2
import os,os.path

datasetPath = "/home/eleganzit/Desktop/maruti/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
samplePath = datasetPath = "/home/eleganzit/Desktop/maruti/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001"
dPath = os.listdir(datasetPath)	# List of all the folder in the train folder
dPath = sorted(dPath)[2:] # Sort and remove the ._.DS_Store, .DS_Strore folder

print datasetPath
print dPath
subfolders = []
sampleFolders = ['optframe', 'optgrayframe', 'wincolor', 'wingray']

for 