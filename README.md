![Profile Views](https://komarev.com/ghpvc/?username=nabulago&color=green)
## Detecting anomalous events in videos by learning deep representations of appearance and motion

An implementation of paper **Detecting anomalous events in videos by learning deep representations of appearance and motion** on python, opencv and tensorflow. This paper uses the _stacked denoising autoencoder_ for the the _feature_ training on the _appearance_ and _motion flow_ features as input for different window sizes and using _multiple SVM_ as a weighted single classifier.

### File details
Most of the files contains the script and details in the files. Once scpit splices the imges of different size for appearance model: windows size - _15x15_, _18x18_, 20x20
Denoising auto encoder file to train the model from the pickle file where you have created the dataset from the images.

##Added new file with include architecture basic in pytorch with new implementation called *main.py* file
*windows size* - _15x15_, _17x17_, 21x21
