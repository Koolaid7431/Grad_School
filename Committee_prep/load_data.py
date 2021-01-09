'''
PhysioNet 2020 challenge

Reads and loads data into a training and test set for TensorFlow as a 2D matrix:
13 rows for 12 leads signals. Lead numbers:  [0,4,10,11,1,2,5,6,7,8,9,3,4] based on ...
15 rows for 12 leads signals. Lead numbers:  [0,4,10,11,6,7,8,4,5,4,7,0,4,2,0]
35 rows for 12 leads signals. Lead numbers:  [0,4,10,11,6, 4,5,6,7,8, 4,7,0,1,2, 6,7,8,10,11, 0,1,6,7,8, 3,4,5,8,9, 3,4,5,1,2]
2500 columns for 5 secend signal at 500Hz

functions used:
    select_win(): To find a start-point of a 5 secend window (starts from R-Peak) 

MRH July 30, 2020 for 2D input and selecting several 5-sec window
Edited Sept 16, 2020
'''

import os
import random
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from select_win import select_win,select_windows

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

tf.random.set_seed(1332) # Set seed for reproducibility

def load_challenge_data(filename): # This function was borrowed from PhysioNet challenge
	x = loadmat(filename)
	data = np.asarray(x['val'], dtype=np.float64)

	new_file = filename.replace('.mat','.hea')
	input_header_file = os.path.join(new_file)

	with open(input_header_file,'r') as f:
		header_data=f.readlines()

	return data, header_data

# Find unique number of classes  
def get_classes(input_directory,files):

	classes=set()
	for f in files:
		g = f.replace('.mat','.hea')
		input_file = os.path.join(input_directory,g)
		with open(input_file,'r') as f:
			for lines in f:
				if lines.startswith('#Dx'):
					tmp = lines.split(': ')[1].split(',')
					for c in tmp:
						classes.add(c.strip())

	return sorted(classes)


def load_data2D_bin(input_directory,win_size=5,fs=500):
	'''
	Input Arguements:
		directory = Full path to the folder containing dataset, containing the .mat and .hea files
		win_size = size of window for selecting a part of signal
		fs = frequency of signals. It is 500Hz for PhysioNet 2020 signals
	Output:
		returns numpy arrays of train_data,train_labels_bin,val_data,val_labels_bin (labels as a list of binary!)
	The code from Calvin creats the filenames in the order of their name, but here I used listdir for the list of files
	'''
	input_files = []
	for f in os.listdir(input_directory):
		if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
			input_files.append(f)
	#random.shuffle(input_files) # Randomly shuffle the list of input files
	num_files = len(input_files)
	print('total number of files: ', num_files)
	classes = get_classes(input_directory,input_files)
	num_classes = len(classes)

	# dividing dataset to train/val/test subsets (80/10/10)

	train_split = int(num_files*0.8) 
	val_split = int(num_files*0.1)
	test_split = int(num_files*0.1)
	print('Number of files for trainig: ', train_split)
	print('Number of files for validation: ', val_split)
	print('Number of files for testing: ', test_split)

	# Lists to hold data and labels
	train_data = []
	train_labels = []
	val_data = []
	val_labels = []

	# Create Training Data and Labels
	for file_number in range(train_split):
		if (file_number+1)%500 == 0:
			print('{} files were loaded for training!'.format(file_number+1))

		# Load mat file for data
		tmp_input_file = os.path.join(input_directory,input_files[file_number])
		data,header_data = load_challenge_data(tmp_input_file)
		labels = header_data[15].split()[1].split(',')		# find all labels (classes) of the signal
			
		# The following code used a new function "select_windows" instead of "select_win"        
		# "select_windows" returns more 5-sec windows depend on the length of signal and the label of the signal        
		# It use sliding window on the signal by stride of 1, 3, or 7 seconds (each second is 500 samples),        
		# so the 5-sec signals will have overlap for some of the signals (minority classes)       
		windows_start = select_windows(input_directory, input_files[file_number], win_size)

		# This loop read all 12 leads one by one and put them together into "window" list, then adds "window" to the train_data
		for win_start in windows_start:
			window = []
			for j in [0,1,2,3,4,5,6,7,8,9,10,11]: # len(data) is 12 for 12-lead signals
			#for j in [0,4,10,11,1,2,5,6,7,8,9,3,4]: # len(data) is 13 for 12-lead signals
			#for j in [0,4,10,11,6,7,8,4,5,4,7,0,4,2,0]: # len(data) is 15 for 12-lead signals
				window.append(data[j][win_start:win_start+win_size*fs])
			# add window to the train dataset
			train_data.append(window)
			# Some signals are belong to more than one class, so we add a list of labels for each window
			# add label_list to the train labels
			train_labels.append(labels)

	# Repeat the above code for validation data
	for file_number in range(train_split,num_files - test_split):
		if (file_number+1)%500 == 0:
			print('{} files were loaded for validation!'.format(file_number+1-train_split))

		# print('Create val_Data: File number = {} and file name = {}'.format(file_number, file_name))
		# Load mat file for data
		tmp_input_file = os.path.join(input_directory,input_files[file_number])
		data,header_data = load_challenge_data(tmp_input_file)
		labels = header_data[15].split()[1].split(',')		# find all labels of the signal

		# The following code used a new function "select_windows" instead of "select_win"        
		# "select_windows" returns more 5-sec windows depend on the length of signal and the label of the signal        
		# It use sliding window on the signal by stride of 1, 3, or 7 seconds (each second is 500 samples),        
		# so the 5-sec signals will have overlap for some of the signals (minority classes)       
		windows_start = select_windows(input_directory, input_files[file_number], win_size)

		# This loop read all 12 leads one by one and put them together into "window" list, then adds "window" to the train_data
		for win_start in windows_start:
			window = []
			for j in [0,1,2,3,4,5,6,7,8,9,10,11]: # len(data) is 12 for 12-lead signals
			#for j in [0,4,10,11,1,2,5,6,7,8,9,3,4]: # len(data) is 13 for 12-lead signals
			#for j in [0,4,10,11,6,7,8,4,5,4,7,0,4,2,0]: # len(data) is 15 for 12-lead signals
				window.append(data[j][win_start:win_start+win_size*fs])
			# add window to the test dataset
			val_data.append(window)
            
			# Some signals are belong to more than one class, so we make a list of labels (1 for occuring lable)
			# add label_list to the val labels
			val_labels.append(labels)

	# Fit the multi-label binarizer on the training set
	print("Labels:")
	mlb = MultiLabelBinarizer()
	mlb.fit(train_labels)

	# Loop over all labels and show them
	N_LABELS = len(mlb.classes_)
	for (i, label) in enumerate(mlb.classes_):
		print("{}. {}".format(i, label))
    
	# transform the targets of the training 
	train_labels_bin = mlb.transform(train_labels)
	# transform the targets of the validation 
	val_labels_bin = mlb.transform(val_labels)

	c = 0
	with open('output_test_files_00.txt', 'w') as f:
		for file_number in range(train_split+val_split,num_files):
			c += 1
			f.write(input_files[file_number]+ '\n')
	print('\nnumber of test signals = ', c) 



	print('\ntrain_data type:',  type(train_data))
	print('train_labels  type:',  type(train_labels))
	print('val_data  type:',  type(val_data))
	print('val_labels  type:', type(val_labels))


	print('\ntrain_data length', len(train_data))
	print('train_labels length', len(train_labels))
	print('val_data length:', len(val_data))
	print('val_labels length:', len(val_labels))

	# Convert to numpy array since TF likes those better than lists
	train_data = np.asarray(train_data,dtype=np.float64)
	val_data = np.asarray(val_data,dtype=np.float64)

	print('\ntrain_data type:',  type(train_data))
	print('train_labels_bin  type:',  type(train_labels_bin))
	print('val_data  type:',  type(val_data))
	print('val_labels_bin  type:', type(val_labels_bin))

	print('\ntrain_data.shape:', train_data.shape)
	print('train_labels.shape:', train_labels_bin.shape)
	print('val_data.shape:', val_data.shape)
	print('val_labels.shape:', val_labels_bin.shape)
	
	# reshape train and validation data to 2D for 2D CNN
	dim = train_data.shape
	train_data = train_data.reshape(dim[0],dim[1],dim[2],1)
	dim = val_data.shape
	val_data = val_data.reshape(dim[0],dim[1],dim[2],1)

	print('\ntrain_data.shape:', train_data.shape)
	print('val_data.shape:', val_data.shape)

	return(train_data,train_labels_bin,val_data,val_labels_bin)

def load_tf(train_data,train_labels_bin,val_data,val_labels_bin):
	'''
	Input Arguements:
	train_data = numpy array of training data
	train_labels_bin = numpy array of training labels (binary list for multi class)
	val_data = numpy array validation data
	val_labels_bin = numpy array of validation labels (binary list for multi class)

	returns tf datasets for use directly in training
	
	***
	In future this function may merge with load_data
		- Kept separate for now incase there is different preprocessing people want
	***
	'''

	train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels_bin))
	val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels_bin))

	BATCH_SIZE = 64 # Training batch size 
	SHUFFLE_BUFFER_SIZE = 2048 # Determines randomness

	# Create training and validation batches

	train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
	# val_dataset = val_dataset.batch(BATCH_SIZE)
	# ----->>>>>> Shuffle validation data too!
	val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

	print('\ntrain_dataset type: ', type(train_dataset))
	print('val_dataset type: ', type(val_dataset))
	return train_dataset,val_dataset

'''
Testing with the Code below
'''

if __name__=='__main__':
	train_data,train_labels,val_data,val_labels = load_data("Training_WFDB")
	train_dataset,val_dataset=load_tf(train_data,train_labels,val_data,val_labels)
