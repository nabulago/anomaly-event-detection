import cv2
import numpy as np


file1 = cv2.imread("/home/eleganzit/Desktop/maruti/cropped/8_009.jpg",0)
file2 = cv2.imread("/home/eleganzit/Desktop/maruti/cropped/9_009.jpg",0)

#file1 = cv2.cvtColor(file1,cv2.COLOR_BGR2GRAY)
#file2 = cv2.cvtColor(file2,cv2.COLOR_BGR2GRAY)


hsvImg = np.zeros_like(file1)
hsvImg[																																																																																																																																																																																																																																																																																																					1] = 255 
# cv2.startWindowThread()
# cv2.namedWindow('Previous Frame',1)
cv2.imshow('Previous Frame',file1)
# cv2.startWindowThread()
# cv2.namedWindow('Next Frame',2)
cv2.imshow('Next Frame', file2)

flow = cv2.calcOpticalFlowFarneback(file1, file2,  None, 0.5, 3, 15, 3, 5, 1.2, 0)
flow
print file1.shape
print file2.shape

print flow.shape
#Obtain the flow magnitude and direction angle
print flow[...,0]
print flow[...,1]
mag, ang = cv2.cartToPolar( flow[...,0], flow[...,1])
print mag.shape, ang.shape
#print flow
#xFlow, yFlow = flow[..., 0], flow[..., 1]

#print frameShape

# Update the color image
hsvImg[..., 0] = 0.5 * ang * 180 / np.pi
# hsvImg[..., 1] = 255
# hsvImg[..., 2] = cv2.normalize(mag, None, 0, 256, cv2.NORM_MINMAX)
# rgbImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)

if cv2.waitKey(0) & 0xff == ord('q'):
	cv2.destroyAllWindows()	
print "end"