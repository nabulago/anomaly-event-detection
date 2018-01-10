'''
    This create the dataset for the flatten, normalize and stack
    the images of motion features which are gray
'''
import cv2
import numpy as np
import pickle
import os
from random import shuffle
trainfolders = ['Train001', 'Train006', 'Train011', 'Train016', 'Train021', 'Train026', 'Train031',
'Train002', 'Train007', 'Train012', 'Train017', 'Train022', 'Train027', 'Train032',
'Train003','Train008', 'Train013', 'Train018', 'Train023', 'Train028', 'Train033',
'Train004', 'Train009', 'Train014', 'Train019', 'Train024', 'Train029', 'Train034',
'Train005', 'Train010', 'Train015', 'Train020', 'Train025', 'Train030']

# trainfolders = sorted(trainfolders)[:10]
fileData = []
flGray = 'wingray' # Folder where the gray motion features are stored
# Path to the training folder as the extracted images are in the train folders
path = '/home/eleganzit/Desktop/maruti/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/'


for tfds in trainfolders:
    flds = str(path) + str(tfds) +'/'+str(flGray)
    print flds
    fileData = fileData + [flds]
# fileData = sorted(fileData)
print len(fileData)
fileData = list(set(fileData))
print len(fileData)
i = 0
# while i<5:
#     print fileData[i]
#     sampleOne = fileData[i]
#     i = i +1
#print "out"
# shuffle(fileData)
filesAll = []
cnt = 20
images = np.ones((225,),dtype='float')

with open("temporary.p",'wb') as imageDataset:
    for fdt in fileData:
        for fls in os.listdir(fdt):
            fl1  = str(fdt)+"/"+str(fls)
            print fl1 
            filesAll = filesAll + [fl1]
            # if cnt < 0:
            #     break
            # cnt = cnt - 1
            # print "data shuffled"
            # print filesAll
            # i = 0
            # print type(images)
            # while i<len(fi):
                # print "Second loop"
                # print fileData[i]
                # sampleOne = fileData[i]
                # fileShow = datasetPath+"/"+sampleOne
                # print fileShow
            
            ########################## Now
            image = cv2.imread(fl1,0)
            image = cv2.resize(image, (15, 15))
            ###########################

            cv2.imshow("image",image)
            image.shape
            # print "image shape" + str(image.flatten().shape)
            # print type(image)
            
            ########### now
            image = image.flatten()
            ############ now

            # cv2.imshow("images",images)
            
            ########### now
            image = cv2.normalize((image).astype('float'),  None,  0,  1,  cv2.NORM_MINMAX)
            ########### now

            #print image
            # print "After normalization" +str(image.shape)
            
            ########### now
            images = np.column_stack((images,image))
            ########### now
            
            #print type(images)
            # print images.shape
            # print images
            
            ########### now
            print "Images : " + str(images.shape[1])
            ########### now


            # print len(images)
            # k = cv2.waitKey(1) & 0xff
            # if k == 27:
            #     cv2.destoyAllWindows()
            #     break
            # elif k == ord('q'):
            #     cv2.destoyAllWindows()
            #     break
            pickle.dump(image,imageDataset,-1)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                pickle.dump(images,imageDataset,-1)
                imageDataset.close()
                print imageDataset.closed
                if not imageDataset.closed:
                    imageDataset.close()
                cv2.destroyAllWindows()
                break
            elif k == ord('q'):
                pickle.dump(images,imageDataset,-1)
                imageDataset.close()
                print imageDataset.closed
                if not imageDataset.closed:
                    imageDataset.close()
                cv2.destroyAllWindows()
                break
            i = i +1
            if i == 5000:
                break
            # print "Images : " +str(i)
        
        # pickle.dump(images,imageDataset,-1)
if not imageDataset.closed:
    pickle.dump(images,imageDataset,-1)
    imageDataset.close()