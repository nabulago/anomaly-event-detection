# Dense optical flow
# https://gist.github.com/myfavouritekk/2cee1ec99b74e962f816

"""
	This file is used to extract motion features from
	the images and not from the video.
	The motion features are in color and gray scale format
	The gray frame is used as the motion features
	not the motion vectors of x and y direction
	each frame in color and gray scale,
	every block from each frame in color and gray
	are stored in separate folder
	Change the directories to save in the path
	Provide the training folder path
	Dependency on the Tkinter package for the folder selection dialog
	If not installed please install Tkinter for version less than 3
	for 
	Some functions the imutils package is used please install it before running the script

"""
import numpy as np
import cv2
import os
import os.path
import argparse
import imutils
import time
import mylib
# Python version 2.7
import Tkinter, tkFileDialog
# Python Version 3.0

root = Tkinter.Tk ()
dirname = tkFileDialog.askdirectory (parent=root, initialdir="~/$USER/Desktop", title='Please select a directory')
if len (dirname) > 0:
    print "You chose %s" % dirname



winW,  winH = 15, 15

dPath = os.listdir (dirname)

# Or You can get all the files path using this line
# This block of code is working to collect files
# listOfFiles = []
# for rootFolder, allFolders, allFiles in os.walk(datapath):
# 	for fldr in allFolders:
# 		for fls in allFiles:
# 			fullFileName = os.path.join(rootFolder,fldr,fls)
# 			listOfFiles.append(str(fullFileName))

# Folders to exclude if are generated using the other script
# These are the folder for the extracted patches and frames
# optframe: optical flow color frame,
# optgrayframe: optical flow gray frames,
# wincolor: extracted patch in color,
# wingray: extracted patch in grayscale

listtoremove = ['._.DS_Store', '.DS_Store', 'optframe', 'optgrayframe', 'wincolor','wingray']

for fd in dPath:
    # Here we will list all the folders that have the frames
	if fd in listtoremove:
		dPath.remove(fd)
dPath = sorted(dPath) # Sort and remove the ._.DS_Store,  .DS_Strore folder

print dPath

# Sample path is taken but not used anywhere in this script
path = "/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001"

windSizes = ['15', '18', '20', 'rs18', 'rs20']
frameSizes = [[240, 165], [252, 162], [240, 160]]
# This path is used for the script
folders = dPath
# files = os.listdir(path)
# files = sorted(files)[2:]
# print files

# Start the webcam
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('../data/vtest.avi')

# Take the first frame and convert it to gray
# ret,  frame = cap.read()

#index = 0

Divisor = 255  # For the grid of the vectors
#print "Type of : " + str(type(dPath))
for idx, f in enumerate (dPath):
    print "Folder: " + str (f)
    # print os.path.join(str(datasetPath), str(f))
    print os.path.join (str (dirname), str (f))
    # files = sorted (os.listdir (os.path.join (str (datasetPath), str (f))))
    files = sorted (os.listdir (os.path.join (str (dirname), str (f))))

    if '.DS_Store' in files:
		files.remove('.DS_Store')
    if '._.DS_Store' in files:
        files.remove ('._.DS_Store')
    if 'wingray' in files:
        files.remove ('wingray')
    if 'optframe' in files:
        files.remove ('optframe')
	if 'wincolor' in files:
		files.remove('wincolor')
    if 'optgrayframe' in files:
        files.remove ('optgrayframe')

    for flp in files:
        # print flp
        if flp in listtoremove:
            files.remove (flp)

	#print listOfFiles
    # print files
#	index = 0
    print "Length of file list: " + str (len (files))
    for index, file in enumerate (files):
        # print "Full File Path: " + os.path.join(str(datasetPath), str(f), str(files[index]))
		#print "Full File Path: " + os.path.join(str(datasetPath), str(f), str(files[index+1]))
		# #prin	t index
		# # Create the HSV color image
		#if len(files)>=199: break
		# print index
		# print index+1
		# print files[index]
		# print files[index+1]
        if index <= 198:
            prevFrameName = files[index]
            nextFrameName = files[index + 1]
            # print
            # print files[index+1]
            # print file
            # print os.path.join(str(datasetPath), str(f), str(files[index]))
            # print os.path.join(str(dPath), str(f), str(files[index]))


            print (os.path.join (str (dirname), str (f), str (files[index])))
            prevFrame = cv2.imread (os.path.join (str (dirname), str (f), str (files[index])))
            cv2.namedWindow ('Prev Frame')
            cv2.imshow ('Prev frame', prevFrame)

            if os.path.isfile (os.path.join (str (dirname), str (f), str (files[index]))):
                print ("File found: " + str (os.path.join (str (dirname), str (f), str (files[index]))))
                # print (os._exists(os.path.join(str (dirname), str (f), str (files[index]))))
                prevFrame = cv2.imread (os.path.join (str (dirname), str (f), str (files[index])))
            else:
                print ("File not found: " + str (str (os.path.join (str (dirname), str (f), str (files[index])))))
                pass

            #print index+1
            if os.path.isfile (os.path.join (str (dirname), str (f), str (files[index + 1]))):
                print ("File found: " + str (os.path.join (str (dirname), str (f), str (files[index + 1]))))
                nextFrame = cv2.imread (os.path.join (str (dirname), str (f), str (files[index + 1])))
            else:
                print ('File not found:')
                pass

            prevFrame =  cv2.resize(prevFrame, (240, 165), 1)
            nextFrame =  cv2.resize(nextFrame, (240, 165), 1)
            mainFrame = prevFrame.copy()
            prevGray = cv2.cvtColor(prevFrame,  cv2.COLOR_BGR2GRAY)
            nextGray = cv2.cvtColor(nextFrame,  cv2.COLOR_BGR2GRAY)

            hsvImg = np.zeros_like(prevFrame)
            hsvImg[..., 1] = 255
            cv2.startWindowThread()
            cv2.namedWindow('Previous Frame', 1)
            cv2.imshow('Previous Frame', prevFrame)
            cv2.startWindowThread()
            cv2.namedWindow('Next Frame', 1)
            cv2.imshow('Next Frame',  nextFrame)

            flow = cv2.calcOpticalFlowFarneback(prevGray,  nextGray,  None,  0.5,  3,  15,  3,  5,  1.2,  0																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																										)
            #Obtain the flow magnitude and direction angle
            mag,  ang = cv2.cartToPolar(flow[...,  0],  flow[...,  1])
            #print flow
            xFlow,  yFlow = flow[...,  0],  flow[...,  1]
            xFlow, yFlow = flow[..., 0], flow[..., 1]
            # Normalize horizontal and vertical components
            horz = cv2.normalize (flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
            vert = cv2.normalize(flow[..., 1],  None,  0,  255,  cv2.NORM_MINMAX)

            horz = horz.astype('uint8')
            vert = vert.astype('uint8')
            # Show the components as images
            cv2.imshow('Horizontal Component',  horz)
            cv2.imshow('Vertical Component',  vert)

            #print frameShape

            # Update the color image
            hsvImg[...,  0] = 0.5 * ang * 180 / np.pi
            hsvImg[...,  1] = 255
            hsvImg[...,  2] = cv2.normalize(mag,  None,  0,  256,  cv2.NORM_MINMAX)
            rgbImg = cv2.cvtColor(hsvImg,  cv2.COLOR_HSV2BGR)
            motionGray = cv2.cvtColor(rgbImg,  cv2.COLOR_BGR2GRAY)

            motionGray1 = cv2.cvtColor(motionGray,  cv2.COLOR_GRAY2BGR)
            # Display the resulting frame
            # cv2.startWindowThread()
            # cv2.namedWindow('Dense optical flow', 1)
            cv2.imshow('Dense optical flow',  np.hstack((prevFrame,  rgbImg, motionGray1)))
            #cv2.imshow('Maked image',  cv2.bitwise_or(mainFrame, mainFrame, mask=motionGray))
            tmppath = os.path.join (str (dirname), str (f))
            if not os.path.exists(os.path.join(tmppath, "optframe")):
                os.makedirs(os.path.join(tmppath, "optframe"))
            if not os.path.exists(os.path.join(tmppath, "optgrayframe")):
                os.makedirs(os.path.join(tmppath, "optgrayframe"))
            cv2.imwrite(str(tmppath)+"/optframe/"+str(index+1)+"_"+str(file.split('.')[0])+".jpeg", rgbImg)
            cv2.imwrite(str(tmppath)+"/optgrayframe/"+str(index+1)+"_"+str(file.split('.')[0])+".jpeg", motionGray)

            # Here display the optical flow in the image
            mylib.dispOpticalFlow (prevFrame, flow, 10, 'Optical flow')

            winNo = 0

            rgbImg = cv2.resize(rgbImg, (240, 165), 1)
            motionGray = cv2.resize(motionGray, (240, 165), 1)
            for (x, y, window) in mylib.sliding_window (rgbImg, stepSize=15, windowSize=(15, 15)):
                if not os.path.exists(os.path.join(tmppath, "wincolor")):
                    os.makedirs(os.path.join(tmppath, "wincolor"))
                if window.shape[0] != winH or window.shape[1] != winW:
                    print winNo
                    #cv2.imshow("Internal window", window)
                    continue
                clone = rgbImg.copy()
                clone1 = motionGray.copy()
                cv2.rectangle(clone,  (x,  y),  (x + winW,  y + winH),  (0,  255,  0),  2)
                cv2.rectangle(clone1,  (x,  y),  (x + winW,  y + winH),  (0,  255,  0),  2)

                cv2.startWindowThread ()
                cv2.namedWindow ("Clone", 1)
                cv2.imshow("Clone",  clone)
                cv2.startWindowThread ()
                cv2.namedWindow ("Window", 1)
                cv2.imshow("Smaller window", window)
                #print str(tmppath)+"/wincolor/"+str(winNo)+"_"+str(file.split('.')[0])+".jpeg"#, window
                # tmppth is the pth to the training folder' Train folder full path
                # Wincolor : the folder name - color patch folder
                # winno : patch no in the particular frame
                # file.split will get the frame no with jpeg extension added to it
                # Change the path accordingly where you want to save in the make directory lines
                cv2.imwrite(str(tmppath)+"/wincolor/"+str(winNo)+"_"+str(file.split('.')[0])+".jpeg", window)
                winNo = winNo + 1
                k1 = cv2.waitKey(1) & 0xFF
                if k1==27:
                    cv2.destroyAllWindows()
                    break
                elif k1 == ord('q'):
                    cv2.destroyAllWindows()
                    break
                elif k1 == ord('e'):
                    cv2.destroyAllWindows()
                    os._exit(0)
                #time.sleep(0.025)
            # Motion gray splicing
            winNo1 = 0
            motionGray = cv2.resize(motionGray, (240, 165), 1)
            for (x1, y1, window1) in mylib.sliding_window (motionGray, stepSize=15, windowSize=(15, 15)):
				if not os.path.exists(os.path.join(tmppath, "wingray")):
					os.makedirs(os.path.join(tmppath, "wingray"))
				# print winNo
				# print x, y, window.shape
				# print winH,  winW
				# if the window does not meet our desired window size,  ignore it
				# print "windows pre"
				# print "Windows pre shape : " +str(window1.shape)
				if window1.shape[0] != winH or window1.shape[1] != winW:
					# print window1.shape[0]
					# print window1.shape[1]
					# print winNo1
					#cv2.imshow("Internal window", window)
					# print "window1"
					continue
			# since we do not have a classifier,  we'll just draw the window
				clone1 = motionGray.copy()
				# print "Clone 1 : " + str(clone1.shape)
				cv2.rectangle(clone1,  (x1,  y1),  (x1 + winW,  y1 + winH),  (0,  255,  0),  2)

                #cv2.startWindowThread()
				#cv2.namedWindow("Clone", 1)
				#cv2.startWindowThread()
				#cv2.namedWindow("Window", 1)
				cv2.imshow("Clone gray",  clone1)
				cv2.imshow("Smaller window gray", window1)
				#print str(tmppath)+"/wingray/"+str(winNo1)+"_"+str(file.split('.')[0])+".jpeg"#, window1
				cv2.imwrite(str(tmppath)+"/wingray/"+str(winNo1)+"_"+str(file.split('.')[0])+".jpeg", window1)

				tk1 = cv2.waitKey(1) & 0xFF
				if tk1==27:
					cv2.destroyAllWindows()
					break
				elif tk1 == ord('q'):
					cv2.destroyAllWindows()
					break
				elif tk1 == ord('e'):
					cv2.destroyAllWindows()
					os._exit(0)
					break
				#time.sleep(0.025)
				winNo1 = winNo1 + 1

            k = cv2.waitKey(1) & 0xFF
            if k==27:
                cv2.destroyAllWindows()
                break
            elif k == ord('q'):
                cv2.destroyAllWindows()
                break
            elif k == ord('e'):
                cv2.destroyAllWindows()
                # If problem due to exit syntax change/update accordingly
                os._exit(0)
cv2.destroyAllWindows()
os._exit(0)
