'''
Load_data.py

Authors: Calvin Zhu
Date: September 8, 2020

Last Update: September 16, 2020

Loads images into TF data sets in order to train/test models.
This is meant to be a library of functions, not stand-alone code.

To Do:
- N/A


Version log:
- Sept 8, 2020: File Created
- Sept 16, 2020: Comments / Descriptions Added
'''

'''
Import Libraries
'''

import tensorflow as tf
import pathlib

'''
Image Parameters/Batch Settings
'''

# Resize Images to this size
# Manually adjust depending on which model architecture
# Xception requires >71x71, recommended 150x150 on TF site
# VGG19 requires >32x32, recommended 200x200 on TF site
# ResNet50 requires >32x3x, recommended 200x200 on TF site

IMG_HEIGHT = 224 
IMG_WIDTH = 224

# Choose Batch Size
# TF Defaults to 32

BATCH_SIZE = 32

'''
Functions
'''

def load_data_trv(train_dir):
	'''
	Loads training and validation sets.
	Organize the file directory such that folder names are labels.
	Each folder contains all the data for that label.
	Function will create resized, labelled pairs in batches.

	Arguments:
	train_dir : string which is the directory containing the folders

	Returns:
	train_set : tf.data object containing the training set
	val_set : tf.data object containing the validation set

	'''

	train_dir = pathlib.Path(train_dir)

	train_set = tf.keras.preprocessing.image_dataset_from_directory(
		train_dir,
		validation_split=0.2,
		subset = "training",
		seed = 123,
		image_size = (IMG_HEIGHT, IMG_WIDTH),
		batch_size = BATCH_SIZE,
		)
    
	val_set = tf.keras.preprocessing.image_dataset_from_directory(
		train_dir,
		validation_split=0.2,
		subset = "validation",
		seed = 123,
		image_size = (IMG_HEIGHT, IMG_WIDTH),
		batch_size = BATCH_SIZE,
		)

	return train_set, val_set

def cache_data(train_set,val_set):
	'''
	Caches the training and validation set.
	Run for performance if loading the data multiple times.

	Arguments:
	train_set : tf.data object containing the training set
	val_set : tf.data object containing the validation set

	Returns:
	train_set : tf.data object containing the training set
	val_set : tf.data object containing the validation set	

	'''

	train_set = train_set.cache().prefetch(buffer_size=10)
	val_set = val_set.cache().prefetch(buffer_size=10)

	return train_set, val_set

'''
Main
'''
def main():
	train_set,val_set = load_data_trv("data_09_22_20/sonix_train")

	for image, label in train_set.take(1):
		print("Image shape: ", image.numpy().shape)
		print("Label: ", label.numpy())


'''
Testing
'''

if __name__ == '__main__':
	main()
	print(1)
