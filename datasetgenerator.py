'''
    flatten image, normalize, stack and add to dataset
'''
import cv2
import numpy as np
from os.path import isfile, join
from os import listdir
from random import shuffle
from shutil import copyfile
import os
import pickle
import cv2
datasetPath = '/apperance'
# trainFolders = ['Train001','Train002']

# image_15.txt has the filepaths of all the apperance windows
# of different scales

fileData = open('./apperance/image_15.txt','r').read().split('\n')
fileData = sorted(fileData)
#print len(fileData)
fileData = list(set(fileData))
print len(fileData)
i = 0
# while i<5:
#     print fileData[i]
#     sampleOne = fileData[i]
#     i = i +1
#print "out"
# shuffle(fileData)       # Don't shuffle the data if you want to consecutive samples
#print "data shuffled"
# i = 0
images = np.ones((225,),dtype='uint8')
print type(images)
print fileData[10]

for i,flD in enumerate(fileData):
    print "File data : " + str(i) +str(fileData[i])
    if i==20:
        break

with open("apperance_main.pkl",'wb') as imageDataset:
    for flD in fileData:
        if i%100 == 0:
            print "===================="
            print "Completed 100 images"
            print "===================="
            print "Total images saved : " + str(i)
        if i%1000 == 0:
            print "===================="
            print "Completed 1000 images"
            print "===================="
            print "Total images saved : " + str(i)

        if i > 20000:
            print "===================="
            print "Completed 20000 files"
            print "===================="
            break
        # print "Second loop"
        print flD
        # print fileData
        # print "File data : " +str(fileData[i])
        # sampleOne = fileData[i]
        # fileShow = datasetPath+"/"+sampleOne
        fileShow = datasetPath+"/"+str(fileData[i])
        # print fileShow
        image = cv2.imread(fileShow,0)
        # if not image.any(): continue
        # print image.shape
        image = cv2.resize(image, (15,15),0)
        #cv2.imshow("image",image)
        image.shape
        # print "image shape" + str(image.flatten().shape)
        # print type(image)
        image = image.flatten()
        #cv2.imshow("images",images)
        image = cv2.normalize((image).astype('float'),  None,  0,  1,  cv2.NORM_MINMAX)
        #print image
        # print "After normalization" +str(image.shape)
        images = np.column_stack((images,image))

        #print type(images)
        #print images.shape
        #print images
        # print images.shape
        # k = cv2.waitKey(1) & 0xff
        # if k == 27:
        #     pickle.dump(image,imageDataset,-1)
        #     imageDataset.close()
        #     if not imageDataset.closed:
        #         imageDataset.close()
        #     cv2.destoyAllWindows()
        #     break
        # elif k == ord('q'):
        #     pickle.dump(image,imageDataset,-1)
        #     if not imageDataset.closed:
        #         imageDataset.close()
        #     cv2.destoyAllWindows()
        #     break

        # k = cv2.waitKey(0) & 0xff
        # if k == 27:
        #     cv2.destoyAllWindows()
        #     break
        # elif k == ord('q'):
        #     cv2.destoyAllWindows()
        #     break
        i = i +1
        pickle.dump(image,imageDataset,-1)
        # if not imageDataset.closed:
        #     pickle.dump(images,imageDataset,-1)
        #     imageDataset.close()
    if not imageDataset.closed:
        imageDataset.close()
    #imd = open('motionfeatures.p','rb')
    #imdataset = pickle.load(imd)
