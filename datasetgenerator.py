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

# def seperateData(data_dir):
#     for filename in listdir(data_dir):
#         if isfile(join(data_dir, filename)):
#             tokens = filename.split('.')
#             if tokens[-1] == 'jpg':
#                 image_path = join(data_dir, filename)
#                 if not os.path.exists(join(data_dir, tokens[0])):
#                     os.makedirs(join(data_dir, tokens[0]))
#                 copyfile(image_path, join(join(data_dir, tokens[0]), filename))
#                 os.remove(image_path)



# class DataSetGenerator:
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         self.data_labels = self.get_data_labels()
#         self.data_info = self.get_data_paths()

#     def get_data_labels(self):
#         data_labels = []
#         for filename in listdir(self.data_dir):
#             if not isfile(join(self.data_dir, filename)):
#                 data_labels.append(filename)
#         return data_labels

#     def get_data_paths(self):
#         data_paths = []
#         for label in self.data_labels:
#             img_lists=[]
#             path = join(self.data_dir, label)
#             for filename in listdir(path):
#                 tokens = filename.split('.')
#                 if tokens[-1] == 'jpg':
#                     image_path=join(path, filename)
#                     img_lists.append(image_path)
#             shuffle(img_lists)
#             data_paths.append(img_lists)
#         return data_paths

#     # to save the labels its optional incase you want to restore the names from the ids 
#     # and you forgot the names or the order it was generated 
#     def save_labels(self, path):
#         pickle.dump(self.data_labels, open(path,"wb"))

#     def get_mini_batches(self, batch_size=10, image_size=(200, 200), allchannel=True):
#         images = []
#         labels = []
#         empty=False
#         counter=0
#         each_batch_size=int(batch_size/len(self.data_info))
#         while True:
#             for i in range(len(self.data_labels)):
#                 label = np.zeros(len(self.data_labels),dtype=int)
#                 label[i] = 1
#                 if len(self.data_info[i]) < counter+1:
#                     empty=True
#                     continue
#                 empty=False
#                 img = cv2.imread(self.data_info[i][counter])
#                 img = self.resizeAndPad(img, image_size)
#                 if not allchannel:
#                     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                     img = np.reshape(img, (img.shape[0], img.shape[1], 1))
#                 images.append(img)
#                 labels.append(label)
#             counter+=1

#             if empty:
#                 break
#             # if the iterator is multiple of batch size return the mini batch
#             if (counter)%each_batch_size == 0:
#                 yield np.array(images,dtype=np.uint8), np.array(labels,dtype=np.uint8)
#                 del images
#                 del labels
#                 images=[]
#                 labels=[]

#     def resizeAndPad(self, img, size):
#         h, w = img.shape[:2]

#         sh, sw = size
#         # interpolation method
#         if h > sh or w > sw:  # shrinking image
#             interp = cv2.INTER_AREA
#         else: # stretching image
#             interp = cv2.INTER_CUBIC

#         # aspect ratio of image
#         aspect = w/h

#         # padding
#         if aspect > 1: # horizontal image
#             new_shape = list(img.shape)
#             new_shape[0] = w
#             new_shape[1] = w
#             new_shape = tuple(new_shape)
#             new_img=np.zeros(new_shape, dtype=np.uint8)
#             h_offset=int((w-h)/2)
#             new_img[h_offset:h_offset+h, :, :] = img.copy()

#         elif aspect < 1: # vertical image
#             new_shape = list(img.shape)
#             new_shape[0] = h
#             new_shape[1] = h
#             new_shape = tuple(new_shape)
#             new_img = np.zeros(new_shape,dtype=np.uint8)
#             w_offset = int((h-w) / 2)
#             new_img[:, w_offset:w_offset + w, :] = img.copy()
#         else:
#             new_img = img.copy()
#         # scale and pad
#         scaled_img = cv2.resize(new_img, size, interpolation=interp)
#         return scaled_img


# if __name__=="__main__":
datasetPath = '/home/eleganzit/Desktop/maruti/apperance'
# seperateData("apperance")
# dsg = DataSetGenerator('/home/eleganzit/Desktop/maruti/apperance')

# trainFolders = ['Train001','Train002']

# image_15.txt has the filepaths of all the apperance windows
# of different scales

## The main logic of the file is here
## The code above the defined is not used
## 
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