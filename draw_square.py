'''
	This file has the code for drawing the ROI
	It will draw the ROI as transparent window with border
	use the function drawROI if you want to use elsewhere
	or copy the function as it is.
	Here is the function with the sample code give 
	which will draw two ROI going in different direction
	on draw no funcitonality added.
'''


import cv2
import numpy as np

def drawROI(img, x, y, alp, winsz, color):
	'''
		It will draw the ROI as transparent window with border
		Args:
			img   : Image which you want to draw the ROI
			x, y  : x and y starting coordinates for the ROI
			alp   : Alpha or transparency value between 0.0 and 1.0
			winsz : Windows size of the ROI winsz x winsz single value
			color : Same color for the border and the filled color in square
		
		Returns:
			Retuns the processed image 
			image = drawROI()
	'''
	ovly = img.copy()
	out = img.copy()

	cv2.rectangle(ovly, (x, y), (x+winsz, y+winsz), (color), -1)
	cv2.rectangle(ovly, (x, y), (x+winsz, y+winsz), (color), 2)
	cv2.addWeighted(ovly, alp, out, 1 - alp, 0,out)

	return out

image = cv2.imread('sampleimg.tif',0)
image1 = cv2.imread('sampleimg.tif',1)
print(image1.shape)
h,w = 158,238
wSz = 20
x, y = 0,0
overlay1 = np.ones((15,15,3))*[0,255,0]
# while (1):
for j in range(0,h,wSz):
	# myy = j+wSz
	# print(myy)
	for i in range(0,w,wSz):
		# myx = i +wSz
		# print(myx)
		imgs = drawROI(image1,i+wSz,j+wSz,0.6,wSz,(255,0,0))
		imgs = drawROI(imgs,j+wSz,i+wSz,0.6,wSz,(0,0,255))
		cv2.imshow('Images',imgs)
		k = cv2.waitKey(10) & 0xFF
		if k == 27:
			cv2.destroyAllWindows()
			break
		elif k == ord('q'):
			cv2.destroyAllWindows()
			break
		

# while(1):
# 	alpha = 0.2
# 	overlay = image1.copy()
# 	cv2.imshow('Over1',overlay1)
# 	output = image1.copy()
# 	cv2.rectangle(overlay, (x, y), (x+wSz, y+wSz),(0, 0, 255), -1)
# 	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0,output)
	
# 	cv2.imshow('Color',image1)
# 	cv2.imshow('Overlay',overlay)
# 	cv2.imshow('Output',output)

	
	
# 	cutout = image1[x:x+wSz,y:y+wSz]
# 	cv2.imshow('Cutout',cutout)	# Overlay image
# 	# cv2.addWeighted(overlay1, alpha, cutout, 1 - alpha, 0,cutout)

# 	x = x + 15
# 	y = y + 15
