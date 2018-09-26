#sampleScript
import cv2
import numpy as np
import os, os.path

def nameSplitter(text):
	''' 
		It will return the original image file name striped with the prefixes
		Image name is the frame no in the folder.
	'''
	splitText = text.split("_")[-1]
	return splitText

def extractFileName(text):
	#print text
	imagePathSplit = text.split('/')
	tmp = text.split('/')[0]
	imageFileName = text.split('/')[2].split('.')[0]
	#return imageFileName
	return imagePathSplit 
	#return [imageNo,tmp,imageFileName]


# Dataset path where training images in format of the pickle file or other sliced image contains
datasetPath = "/data/train/"
# List of folders which you want to use as for now to test the codes is working 2 folders are taken
listOfFolders = ['Train001','Train002']
# List of slice sizes which are 15, 18 and 20 respectively
listOfWindows = ['15','rs18','rs20']
# Sample name for the sliced images
imageList15 = ['0_15_001.jpeg','1_15_001.jpeg']
imageListrs18 = ['90_rs18_15_001.jpeg','91_rs18_15_001.jpeg']
imageListrs18 = ['120_rs20_15_001.jpeg','121_rs20_15_001.jpeg']

# /train/folder/windowsize/imageno_windowsize_image.jpeg
# /test/folder/windowsize/imageno_windowsize_image.jpeg
imageN = 'Train001/15/0_15_001.jpeg'
imageD = 'folder/windowsize/imageno_windowsize_image.jpeg'


listOfImagesAll = []

for lf in listOfFolders:
	for ws in listOfWindows:
		print lf+"/"+ws

#imn, tst, tb = extractFileName(imageD)
#print (imn, tst, tb)
print ("------------")
print (extractFileName(imageN)[2])
print (nameSplitter(extractFileName(imageN)[2]))
