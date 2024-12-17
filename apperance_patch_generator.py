##
# To run from console
# python appearance_path_generator.py -p /UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train -dd Project/data
# or
# python --path /home/hduser/Desktop/Project/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train --datadst /home/hduser/Desktop

# of if you are using the pycharm provide script arguments
# in the ide menu Run--> Edit configurations --> Script paramters
# Add these parameter values
# --path your_path_to_dataset_trainfolder --datadst path_where_you_want_store_spliced_data
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
ap.add_argument ("-p", "--path", required=False, help="Main directory of dataset")
ap.add_argument ("-dd", "--datadst", help="Directory where you want to save dataset")
args = vars (ap.parse_args ())

# (winW,winH) = (20,20)
winSz = [15, 18, 20]  # Thes are the window sizes for sliding window

pyrCnt = 0
imgCount = 0

# for sz in winSz:
#	print "Window size: " + str(sz)
print ("Start script")

imageDir = args["path"]  # specify your path here
dataDest = args["datadst"]

if args["path"] is None:
    imageDir = raw_input("Please enter the path to the dataset:")
    print ("Entered image dataset path:")
    print (imageDir)
    if not os.path.exists(str(imageDir)):
        print ("Directory doen't exists")

print ("Source dataset Directory: " + str (imageDir))



if args["datadst"] is None:
    dataDest = raw_input("Please enter the path where you want to store the processed or spliced images:")
    print (dataDest)
    print ("Entered spliced or porcessed image dataset path:")
    if not os.path.exists(str(dataDest)):
        print ("Directory doen't exists")
dataDest = str (dataDest) + '/appearance_spliced_images'
mylib.createDirectory(dataDest)

print ("Destination directory to store processed dataset: " + dataDest)


image_path_list = []

print "start: "
# create a list all files in directory and
# append files with a valid extension to image_path_list
folders = os.listdir(imageDir)
print("Image directories : " + imageDir)
print("Folders: " + str (sorted (folders)))
folders.remove('.DS_Store')
folders.remove('._.DS_Store')
imageTmp = []

# fObject = open(dataDest+"/"+"imagelist.txt","a+")
# f_main = open( dataDest+"/"+"image_15.txt","a+")
fObject = open (dataDest + "/" + "all_imagelist.txt", 'w')
f_main = open (dataDest + "/" + "all_image_15.txt", 'w')

# fObject = open ("/home/hduser/Desktop/ped2_imagelist.txt", 'w')
# f_main = open ("/home/hduser/Desktop/ped2_image_15.txt", 'w')

appDatasetAll = open(os.path.join(dataDest,"apperance.p") ,'wb')
appDataset15 = open(os.path.join(dataDest,"apperance15.p") ,'wb')
appDataset18 = open(os.path.join(dataDest,"apperance18.p") ,'wb')
appDataset20 = open(os.path.join(dataDest,"apperance20.p") ,'wb')
if not imageDir.startswith ("._.D"):
    for folder in sorted(folders):
        if not folder.startswith ("."):
            mylib.createDirectory(os.path.join(dataDest,folder))

    for folder in sorted (folders):
        print ("folder : " + folder)
        if not folder.startswith ("."):
            print ("Folder : " + str (folder))
            # print "List of files: " + os.listdir(imageDir+"/"+str(folder))
            print ("Fall: " + imageDir + "/" + str (folder))
            for file in sorted(os.listdir (os.path.join(imageDir,folder))):
                # print "File: " +file
                # print file

                if not mylib.checkValidExtension(file):
                    print("Please provide file with proper extensions")
                    continue

                tpath = str (imageDir) + "/" + str (folder) + "/" + file


                # loop through image_path_list to open each image
                for imagePath in image_path_list:
                    print imagePath
                image = cv2.imread (tpath, 0)

                # display the image on screen with imshow()
                # after checking that it loaded
                if image is not None:

                    for sz in winSz:

                        # mylib.createDirectory(os.path.join(dataDest, folder))
                        mylib.createDirectory(os.path.join(dataDest, folder, str(sz)))
                        if not sz == 15:
                            mylib.createDirectory (dataDest + "/" + folder + "/" + str (sz))
                            print ("In the loop of " + str (sz))

                        tz = sz

                        if sz == 15:
                            resized = cv2.resize (image, (240, 165), interpolation=cv2.INTER_LINEAR)
                        # resized = resizeToMainWindowSize(image,240, 165)
                        elif sz == 18:
                            resized = cv2.resize (image, (252, 162), interpolation=cv2.INTER_LINEAR)
                        # resized = resizeToMainWindowSize(image,252, 162)
                        elif sz == 20:
                            resized = cv2.resize (image, (240, 160), interpolation=cv2.INTER_LINEAR)
                        # resized = resizeToMainWindowSize(image,240,160)

                        for (x, y, window) in mylib.sliding_window (resized, stepSize=15, windowSize=(tz, tz)):

                            # if the window does not meet our desired window size, ignore it
                            # if window.shape[0] != winH or window.shape[1] != winW:
                            if window.shape[0] != sz or window.shape[1] != sz:
                                continue
                            clone = resized.copy ()
                            # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                            cv2.rectangle (clone, (x, y), (x + sz, y + sz), (0, 255, 0), 1)

                            #cv2.imshow ("Image", image)
                            #cv2.imshow ("Clone", clone)
                            #cv2.imshow ("window", window)

                            if sz == 15:
                                tmpFileName = str(imgCount)+ "_"+ str(sz) + "_" + file.split ('.')[0] + ".jpeg"
                                nmi = os.path.join(folder,str(sz),tmpFileName)
                                print (nmi)
                                nm1 = os.path.join(dataDest,nmi)
                                fObject.writelines (str(nmi)+ "\n")
                                f_main.writelines (str(nmi)+ "\n")
                                cv2.imwrite (nm1, window)
                                # Flatten the image
                                windowflat = window.flatten()

                                # normalize the images
                                windowflat = cv2.normalize(windowflat.astype(float),windowflat.astype(float), alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
                                print(windowflat)
                                pickle.dump(windowflat,appDatasetAll)
                                pickle.dump (windowflat, appDataset15)

                                imgCount = imgCount + 1

                            if sz == 18:
                                print("In loop of 18")
                                # nm1 = dataDest+"/"+ folder+ "/" + str(sz)+"/" + str(imgCount)+"_"+str(sz)+"_"+file.split('.')[0]+".jpeg"
                                nm2 = dataDest + "/" + folder + "/" + str (sz) + "/" + str (
                                    imgCount) + "_" + str (sz) + "_" + file.split ('.')[0] + ".jpeg"
                                print (nm2)
                                fObject.writelines (
                                    folder + "/" + str (sz) + "/" + str (imgCount) + "_" + str (sz) + "_" +
                                    file.split ('.')[0] + ".jpeg" + "\n")
                                f_main.writelines (
                                    folder + "/" + str (sz) + "/" + str (imgCount) + "_" + str (sz) +"_"+
                                    file.split ('.')[0] + ".jpeg" + "\n")

                                tlp1 = str (sz) + "x" + str (sz) + " Resized Window Frame"
                                resWin = cv2.resize (window, (15, 15), interpolation=cv2.INTER_LINEAR)
                                #cv2.imshow (tlp1, resWin)
                                cv2.imwrite (nm2, resWin)

                                # Flatten the image
                                windowflat = resWin.flatten()

                                # normalize the images
                                windowflat = cv2.normalize (windowflat.astype (float), windowflat.astype (float),
                                                            alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                                print(windowflat)
                                pickle.dump (windowflat, appDatasetAll)
                                pickle.dump (windowflat, appDataset18)


                                imgCount = imgCount + 1

                            if sz == 20:
                                # print "In loop of 20"
                                # nm1 = dataDest+"/"+ folder+ "/" + str(sz)+"/" + str(imgCount)+"_"+str(sz)+"_"+file.split('.')[0]+".jpeg"
                                nm2 = dataDest + "/" + folder + "/" + str (sz) + "/" + str (
                                    imgCount) + "_" + str (sz) + "_15_" + file.split ('.')[0] + ".jpeg"

                                fObject.writelines (
                                    folder + "/" + str (sz) + "/" + str (imgCount) + "_" + str (sz) + "_" +
                                    file.split ('.')[0] + ".jpeg" + "\n")
                                f_main.writelines (
                                    folder + "/"  + str (sz) + "/" + str (imgCount) + "_" + str (sz) + "_" +
                                    file.split ('.')[0] + ".jpeg" + "\n")
                                # print nm1
                                # print nm2
                                tlp = str (sz) + "x" + str (sz) + " Window"
                                cv2.imshow (tlp, window)
                                tlp1 = str (sz) + "x" + str (sz) + " Resized Window Frame"
                                #cv2.imshow (tlp1, resWin)
                                # cv2.imwrite(nm1,window)
                                resWin = cv2.resize (window, (15, 15), interpolation=cv2.INTER_LINEAR)
                                cv2.imwrite (nm2, resWin)

                                # Flatten the image
                                windowflat = resWin.flatten ()

                                # normalize the images
                                windowflat = cv2.normalize (windowflat.astype (float), windowflat.astype (float),
                                                            alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                                pickle.dump (windowflat, appDatasetAll)
                                pickle.dump (windowflat, appDataset20)

                                imgCount = imgCount + 1


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
                            mylib.exitScript()
                            #time.sleep (0.025)
                            # cv2.imshow(imagePath, image)
                            # pyrCnt = 0
                            # pyrCnt = pyrCnt + 1
                            # pyrCnt = 0
                    imgCount = 0
                elif image is None:
                    print ("Error loading: " + imagePath)
                # Segmentation of file ends here
                print "End of File : " + str (file)

                mylib.exitScript ()
                    # cv2.destroyAllWindows()
        print "End of Folder : " + str (folder)
        print "Press q or esc to exit"
        mylib.exitScript ()
fObject.close ()
f_main.close ()
appDatasetAll.close()
appDataset15.close()
appDataset18.close()
appDataset20.close()
print("End of script")
print("Arguments: " + str (args))
