## 
# python oneimage.py -p /UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train -dd Project/data
__author__ = "Maruti Goswami"

# Load images
# Store in array
# Load 10 images
# Sliding window slices
# for now the path is given from the 



import argparse  # For parsing arguments
import cv2
import os, os.path
import imutils
import time
import pickle
import mylib

# import tensorflow as tf
# import autoencoder

# # #################### #
# #   Flags definition   #
# # #################### #
# flags = tf.app.flags
# FLAGS = flags.FLAGS

# # Global configuration
# flags.DEFINE_string('model_name', '', 'Model name.')
# flags.DEFINE_string('dataset', 'mnist', 'Which dataset to use. ["mnist", "cifar10"]')
# flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
# flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')
# flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name.')
# flags.DEFINE_boolean('encode_train', False, 'Whether to encode and store the training set.')
# flags.DEFINE_boolean('encode_valid', False, 'Whether to encode and store the validation set.')
# flags.DEFINE_boolean('encode_test', False, 'Whether to encode and store the test set.')


# # Stacked Denoising Autoencoder specific parameters
# flags.DEFINE_integer('n_components', 256, 'Number of hidden units in the dae.')
# flags.DEFINE_string('corr_type', 'none', 'Type of input corruption. ["none", "masking", "salt_and_pepper"]')
# flags.DEFINE_float('corr_frac', 0., 'Fraction of the input to corrupt.')
# flags.DEFINE_integer('xavier_init', 1, 'Value for the constant in xavier weights initialization.')
# flags.DEFINE_string('enc_act_func', 'tanh', 'Activation function for the encoder. ["sigmoid", "tanh"]')
# flags.DEFINE_string('dec_act_func', 'none', 'Activation function for the decoder. ["sigmoid", "tanh", "none"]')
# flags.DEFINE_string('main_dir', 'dae/', 'Directory to store data relative to the algorithm.')
# flags.DEFINE_string('loss_func', 'mean_squared', 'Loss function. ["mean_squared" or "cross_entropy"]')
# flags.DEFINE_integer('verbose', 0, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
# flags.DEFINE_integer('weight_images', 0, 'Number of weight images to generate.')
# flags.DEFINE_string('opt', 'gradient_descent', '["gradient_descent", "ada_grad", "momentum"]')
# flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
# flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')
# flags.DEFINE_integer('num_epochs', 10, 'Number of epochs.')
# flags.DEFINE_integer('batch_size', 10, 'Size of each mini-batch.')

# assert FLAGS.dataset in ['mnist', 'cifar10']
# assert FLAGS.enc_act_func in ['sigmoid', 'tanh']
# assert FLAGS.dec_act_func in ['sigmoid', 'tanh', 'none']
# assert FLAGS.corr_type in ['masking', 'salt_and_pepper', 'none']
# assert 0. <= FLAGS.corr_frac <= 1.
# assert FLAGS.loss_func in ['cross_entropy', 'mean_squared']
# assert FLAGS.opt in ['gradient_descent', 'ada_grad', 'momentum']


ap = argparse.ArgumentParser (description="Apperance and Motion DeepNet")
ap.add_argument ("-i", "--image", required=False, help="Path to the image")
ap.add_argument ("-p", "--path", required=False, help="Directory or directories of images")
ap.add_argument ("-dd", "--datadst", help="Directory where you want to save dataset")
args = vars (ap.parse_args ())


def createDirectory ( path ):
    if not os.path.exists (path):
        os.mkdir (path)
        print "Directory created : " + str (path)
    else:
        print "Could not create Directory"


def resizeToMainWindowSize ( win, winSize ):
    cv2.resize (win, (winSize, winSize), interpolation=cv2.INTER_CUBIC)
    return win


def resizeToMainWindowSize ( win, winSize ):
    cv2.resize (win, (winSize[0], winSize[1]), interpolation=cv2.INTER_CUBIC)
    return win


def resizeToMainWindowSize ( win, winSizeW, winSizeH ):
    cv2.resize (win, (winSizeW, winSizeH), interpolation=cv2.INTER_CUBIC)
    return win


# (winW,winH) = (20,20)
winSz = [15, 18, 20]  # Thes are the window sizes for sliding window

pyrCnt = 0
imgCount = 0

# for sz in winSz:
#	print "Window size: " + str(sz)
print "Start script"

imageDir = args["path"]  # specify your path here
print "Directory: " + str (imageDir)
dataDest = args["datadst"]
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]  # specify your valid extensions here
valid_image_extensions = [item.lower () for item in valid_image_extensions]
print "start: "
# create a list all files in directory and
# append files with a vaild extention to image_path_list
folders = os.listdir (imageDir)
print "Image directories : " + imageDir
print "Folders: " + str (sorted (folders))

imageTmp = []

# fObject = open(dataDest+"/"+"imagelist.txt","a+")
# f_main = open( dataDest+"/"+"image_15.txt","a+")
fObject = open (dataDest + "/" + "ped2_imagelist.txt", "a+")
f_main = open (dataDest + "/" + "ped2_image_15.txt", "a+")

if not imageDir.startswith ("._.D"):
    for folder in sorted (folders):
        print "folder : " + folder
        if not folder.startswith ("."):
            print "Folder : " + str (folder)
            # print "List of files: " + os.listdir(imageDir+"/"+str(folder))
            print "Fall: " + imageDir + "/" + str (folder)
            for file in os.listdir (imageDir + "/" + str (folder)):
                # print "File: " +file
                # print file
                extension = os.path.splitext (file)[1]
                # print extension
                if extension.lower () not in valid_image_extensions:
                    continue
                tpath = str (imageDir) + "/" + str (folder) + "/" + file
                # print "TPATH : " + str(tpath)
                # image_path_list.append(tpath)
                # image_path_list.append(os.path.join(imageDir, file))
                # print tpath
                # print image_path_list


                # loop through image_path_list to open each image
                for imagePath in image_path_list:
                    print imagePath

                # fileList = ['001.tif','002.tif','003.tif']
                # print image_path_list


                # imagePath = tpi
                image = cv2.imread (tpath, 0)

                # display the image on screen with imshow()
                # after checking that it loaded
                if image is not None:

                    for sz in winSz:

                        createDirectory (dataDest + "/" + folder)
                        createDirectory (dataDest + "/" + folder + "/" + str (sz))
                        if not sz == 15:
                            createDirectory (dataDest + "/" + folder + "/rs" + str (sz))
                            print "In the loop of " + str (sz)

                        tz = sz

                        if sz == 15:
                            resized = cv2.resize (image, (240, 165), interpolation=cv2.INTER_CUBIC)
                        # resized = resizeToMainWindowSize(image,240, 165)
                        elif sz == 18:
                            resized = cv2.resize (image, (252, 162), interpolation=cv2.INTER_CUBIC)
                        # resized = resizeToMainWindowSize(image,252, 162)
                        elif sz == 20:
                            resized = cv2.resize (image, (240, 160), interpolation=cv2.INTER_CUBIC)
                        # resized = resizeToMainWindowSize(image,240,160)

                        for (x, y, window) in mylib.sliding_window (resized, stepSize=tz, windowSize=(tz, tz)):

                            # if the window does not meet our desired window size, ignore it
                            # if window.shape[0] != winH or window.shape[1] != winW:
                            if window.shape[0] != sz or window.shape[1] != sz:
                                continue

                            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
                            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
                            # WINDOW
                            # since we do not have a classifier, we'll just draw the window
                            clone = resized.copy ()
                            # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                            cv2.rectangle (clone, (x, y), (x + sz, y + sz), (0, 255, 0), 1)

                            cv2.imshow ("Image", image)
                            cv2.imshow ("Clone", clone)
                            cv2.imshow ("window", window)

                            # print window.flatten()
                            # print clone[:].shape
                            ### Uncomment to write to folder
                            # nm = "/extimages/"+str(pyrCnt)+"_"+str(imgCount)+"_" + "" +str(sz)+ "x"+str(sz) +"_001.jpeg"
                            # print file.split('.')[0]


                            if sz == 15:
                                nm1 = dataDest + "/" + folder + "/" + str (sz) + "/" + str (imgCount) + "_" + str (
                                    sz) + "_" + file.split ('.')[0] + ".jpeg"
                                fObject.writelines (
                                    folder + "/" + str (sz) + "/" + str (imgCount) + "_" + str (sz) + "_" +
                                    file.split ('.')[0] + ".jpeg" + "\n")
                                f_main.writelines (
                                    folder + "/" + str (sz) + "/" + str (imgCount) + "_" + str (sz) + "_" +
                                    file.split ('.')[0] + ".jpeg" + "\n")
                                cv2.imwrite (nm1, window)
                                imgCount = imgCount + 1

                            if sz == 18:
                                # print "In loop of 18"
                                # nm1 = dataDest+"/"+ folder+ "/" + str(sz)+"/" + str(imgCount)+"_"+str(sz)+"_"+file.split('.')[0]+".jpeg"
                                nm2 = dataDest + "/" + folder + "/" + "rs" + str (sz) + "/" + str (
                                    imgCount) + "_" + str (sz) + "_15_" + file.split ('.')[0] + ".jpeg"
                                fObject.writelines (
                                    folder + "/" + str (sz) + "/" + str (imgCount) + "_" + str (sz) + "_" +
                                    file.split ('.')[0] + ".jpeg" + "\n")
                                f_main.writelines (
                                    folder + "/" + "rs" + str (sz) + "/" + str (imgCount) + "_" + str (sz) + "_15_" +
                                    file.split ('.')[0] + ".jpeg" + "\n")
                                # print nm1
                                # resWin = resizeToMainWindowSize(window,15,15)
                                # print nm2
                                # cv2.imwrite(nm,window)
                                # tlp = str(sz)+"x"+str(sz)+" Window"
                                # cv2.imshow(tlp,window)
                                # cv2.imwrite(nm1,window)
                                tlp1 = str (sz) + "x" + str (sz) + " Resized Window Frame"
                                resWin = cv2.resize (window, (15, 15), interpolation=cv2.INTER_CUBIC)
                                cv2.imshow (tlp1, resWin)
                                cv2.imwrite (nm2, resWin)

                                imgCount = imgCount + 1

                            if sz == 20:
                                # print "In loop of 20"
                                # nm1 = dataDest+"/"+ folder+ "/" + str(sz)+"/" + str(imgCount)+"_"+str(sz)+"_"+file.split('.')[0]+".jpeg"
                                nm2 = dataDest + "/" + folder + "/" + "rs" + str (sz) + "/" + str (
                                    imgCount) + "_" + str (sz) + "_15_" + file.split ('.')[0] + ".jpeg"
                                fObject.writelines (
                                    folder + "/" + str (sz) + "/" + str (imgCount) + "_" + str (sz) + "_" +
                                    file.split ('.')[0] + ".jpeg" + "\n")
                                f_main.writelines (
                                    folder + "/" + "rs" + str (sz) + "/" + str (imgCount) + "_" + str (sz) + "_15_" +
                                    file.split ('.')[0] + ".jpeg" + "\n")
                                # print nm1
                                # print nm2
                                tlp = str (sz) + "x" + str (sz) + " Window"
                                cv2.imshow (tlp, window)
                                tlp1 = str (sz) + "x" + str (sz) + " Resized Window Frame"
                                cv2.imshow (tlp1, resWin)
                                # cv2.imwrite(nm1,window)
                                resWin = cv2.resize (window, (15, 15), interpolation=cv2.INTER_CUBIC)
                                cv2.imwrite (nm2, resWin)
                                imgCount = imgCount + 1

                            # Convert image to vector and normalize 0-1

                            # dae= autoencoder.DenoisingAutoencoder(model_name='dae', n_components=256, enc_act_func='tanh',dec_act_func='none', loss_func='mean_squared', num_epochs=10, batch_size=10,xavier_init=1, opt='gradient_descent', learning_rate=0.01, momentum=0.9, corr_type='none', corr_frac=0., verbose=1, seed=-1)

                            # Now we make a matrix for each image for now only one image
                            # stacking with different windows sizes
                            # For more complexity we can use color image but it will add more data depend on the channles we will be able to estimate

                            # Optical flow images

                            # Combine the images for all the window sizes


                            # print "Image: " + str(imgCount)
                            # Here we will pre-process the image and corrupt it

                            # DAE section
                            #				cv2.imshow("Patch",image[x:(x+winW),y:(y+winH)])
                            ky = cv2.waitKey (1) & 0xff
                            if ky == 27 | ky == 'q':
                                break
                            time.sleep (0.025)
                            # cv2.imshow(imagePath, image)
                            # pyrCnt = 0
                            # pyrCnt = pyrCnt + 1
                            # pyrCnt = 0
                    imgCount = 0
                elif image is None:
                    print ("Error loading: " + imagePath)
                # end this loop iteration and move on to next image

                # wait time in milliseconds
                # this is required to show the image
                # 0 = wait indefinitely
                # exit when escape key is pressed
                # Segmentation of file ends here
                print "End of File : " + str (file)

                outk = cv2.waitKey (1) & 0xff
                if outk == 27 | outk == 'q':
                    break
                    # cv2.destroyAllWindows()
        print "End of Folder : " + str (folder)
        print "Press q or esc to exit"
        k = cv2.waitKey (1) & 0xff
        if k == 27 | k == 'q':
            cv2.destroyAllWindows ()
fObject.close ()
f_main.close ()
print "End of script"
print "Arguments: " + str (args)
