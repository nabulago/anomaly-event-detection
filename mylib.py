# Library for customized functions
import numpy as np
import os, os.path
import cv2
import imutils


# fileName = "/home/hduser/Documents/experiment/vg_text/001.tif"


def checkValidExtension ( fileName ):
    ''' 
        =======================================
        Check for the valid file extension in 
        given file name or string passed
        =======================================


        Arguments:
            fileName: file name or filename string
    '''

    valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", "jpg", "jpeg", "png", "tif",
                              "tiff"]  # specify your valid extensions here
    valid_image_extensions = [item.lower () for item in valid_image_extensions]

    if fileName.__contains__ ('/'):
        extension = fileName.lower ().split ("/")[-1].split ('.')[-1]
    else:
        extension = fileName.lower ().split ('.')[-1]

    # print extension
    if valid_image_extensions.__contains__ (extension):
        print "Contains " + str (extension)
        # return true


# checkValidExtension(fileName)

def pyramid ( image, scale=1.5, minSize=(15, 15) ):
    '''
        ========================
        Returns an image pyramid
        ========================

        Arguments: 
            image: Image file or object
            scale: Decreasing with a specified ratio
            minSize: Minimum image size

        Return:
            image: Sliced image
    '''
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int (image.shape[1] / scale)
        image = imutils.resize (image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image


def sliding_window ( image, stepSize, windowSize ):
    '''
        Splits the image based on the window size and stride length

        Arguments:
            image: Pass image object or the numpy array
            stepSize: Stride length for frame
            windowSize: window size for the patch from image to extract
        Returns: 
            x, y: Corrdinate of the main image with
            image: Cropped image of dimensions with respect to image size
                    and given window size and stride length
    '''
    # slide a window across the image
    for y in xrange (0, image.shape[0], stepSize):
        for x in xrange (0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def createDirectory ( path ):
    '''
        ================================
        Create directory for a give path
        ================================

        Arguments:
            path: Directory name or path of the directory

        Returns:
            No return values
    '''
    if os.path.exists (path):
        pass
    elif not os.path.exists (path):
        os.mkdir (path)
        print "Directory created : " + str (path)
    else:
        print "Could not create Directory"


def resizeToMainWindowSize ( image, winSize ):
    '''
        ==================================================
        Resize the window size for larger than given image
        ==================================================  

        Arguments:
            image: Image you want to resize
            winSize: Window size of the image
        Returns:
            Resize image of given window size 
    '''
    if type (winSize) == int:
        return cv2.resize (image, (winSize, winSize), interpolation=cv2.INTER_CUBIC)
    elif type (winSize) == []:
        return cv2.resize (image, (winSize[0], winSize[1]), interpolation=cv2.INTER_CUBIC)
    elif type (winSize) == ():
        return cv2.resize (image, (winSize), interpolation=cv2.INTER_CUBIC)


def drawRectangleOnImage ( image, x, y, winSize ):
    clone = image.copy ()
    if type (winSize) == int:
        cv2.rectangle (clone, (x, y), (x + winSize[0], y + winSize[1]), (0, 255, 0), 2)
        return clone


def drawROI ( img, x, y, alp, winsz, color ):
    '''
        It will draw the ROI as transparent window with border over image

        Args:
            img   : Image which you want to draw the ROI
            x, y  : x and y starting coordinates for the ROI
            alp   : Alpha or transparency value between 0.0 and 1.0
            winsz : Windows size of the ROI winsz x winsz single value
            color : Same color for the border and the filled color in square

        Returns:
            Retuns the processed image 

            image = drawROI('image.jpg',30,50,,0.6,20,(0,255,0))

    '''
    ovly = img.copy ()
    out = img.copy ()

    # Draw filled rectangle
    cv2.rectangle (ovly, (x, y), (x + winsz, y + winsz), (color), -1) \
        # Drwa line or border for rectangle
    cv2.rectangle (ovly, (x, y), (x + winsz, y + winsz), (color), 2)
    cv2.addWeighted (ovly, alp, out, 1 - alp, 0, out)

    return out
