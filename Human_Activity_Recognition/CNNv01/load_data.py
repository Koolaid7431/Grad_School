'''
HAR Project

Reads and loads data into three sets: training, validation and val set for TensorFlow as a 2D matrix:
3 rows for xyz sensors (accelerometer/Gyroscope of eitehr smartphone or smartwatch)
100 columns for 5 secend signal at 20Hz

functions used:
select_win(): To find a start-point of a 5 secend window 

MRH July 9, 2020
'''

import os
import random
import numpy as np
import tensorflow as tf
#from scipy.io import loadmat
#from select_win import select_win
import matplotlib.pyplot as plt


tf.random.set_seed(1234) # Set seed for reproducibility
fs = 20 # sampling frequency



def read_person_activity(input_directory, file_name):
	full_name = input_directory + '/' + file_name
	activity_dic = {}
	for indx in range(ord('A'),ord('S')+1): # Label for activities are in range 'A' to 'S' (except 'N')
		activity_dic[(chr(indx))] = []
	activity_dic.pop('N') # There is NOT label 'N' in the dataset

	with open(full_name,'r') as f:
		sample = f.readlines() # example of one line: '1600,A,252207666810782,-0.36476135,8.793503,1.0550842;\n'
	sample_len = len(sample)

	for i in range(sample_len): 
		tmp = sample[i].split(',') # e.g. tmp looks like: '1600,A,252207666810782,-0.36476135,8.793503,1.0550842;\n'
		activity_dic[tmp[1]].append([tmp[3], tmp[4], tmp[5].split(';')[0]])
	subject_code = tmp[0]

	return subject_code, activity_dic




def load_data2D(input_directory):
	'''
	Input Arguements:
	directory = Full path to the folder containing dataset, containing the .txt files for all 51 participants
	Output:
	returns numpy arrays of train_data,train_labels,val_data,val_labels
	I used listdir for the list of files which may be not in the order of the subject's code
	'''
	input_files = []
	for f in os.listdir(input_directory):
		if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('txt'):
			input_files.append(f)
	#random.shuffle(input_files)
	num_files = len(input_files)
	print('total number of files: ', num_files)
	actions = ['A','B','C','D','E']
	win_size=5 # 5 second window
	fs = 20 #sampling frequency
	
	# Create train/val/test split (80/10/10)
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
	flag = True
	for file_number in range(train_split):
		# Load txt file for data
		# subj_activities is a dictionary. Each action is a key in the dictionary
		subj_code, subj_activities = read_person_activity(input_directory, input_files[file_number])
		if flag:
			print('{}_ {} file were loaded for training set!'.format(file_number, input_files[file_number]))
					
		for action in actions:
			activity = np.asarray(subj_activities[action],dtype=np.float64)
			if flag:
				print('length of activity data= {}'.format(activity.shape))
				# flag = False
			# win_start = select_win(input_directory, input_files[file_number], win_size)
			win_start = 0
			# This loop read all 3 axis of sensor (x,y,z) one by one and put them together into "window" list, then adds "window" to the train_data
			while win_start+(win_size*fs)<len(activity):
				window = []
				for j in range(3): # signal is from 3 axis x,y,z
					window.append(activity[win_start:win_start+(win_size*fs),[j]])	
					if flag:
						print('win_start and win_size*fs are {} and {}'.format(win_start,win_size*fs))
						print('activity data window= {}'.format(len(activity[win_start:win_start+(win_size*fs),[j]])))
						flag = False
				train_data.append(window)
				indx = actions.index(action) 
				train_labels.append(indx)
				win_start += win_size*fs

	# Repeat the above code for validation data set
	for file_number in range(train_split,num_files - test_split):
		# Load txt file for data
		subj_code, subj_activities = read_person_activity(input_directory, input_files[file_number])
		print('{}_ {} file were loaded for validaton set!'.format(file_number, input_files[file_number]))	
		
		for action in actions:
			activity = np.asarray(subj_activities[action],dtype=np.float64)
			# win_start = select_win(input_directory, input_files[file_number], win_size)
			win_start = 0
			# This loop read all 3 axis of sensor (x,y,z) one by one and put them together into "window" list, then adds "window" to the train_data
			while win_start+(win_size*fs)<len(activity):
				window = []
				for j in range(3): # signal is from 3 axis x,y,z
					window.append(activity[win_start:win_start+(win_size*fs),[j]])		
				val_data.append(window)
				indx = actions.index(action) 
				val_labels.append(indx)
				win_start += win_size*fs

	c = 0
	with open('HAR_test_files_00.txt', 'w') as f:
		for file_number in range(train_split+val_split,num_files):
			c += 1
			f.write(input_files[file_number]+ '\n')
	print('number of test signals = ', c) 

	print('train_data type:',  type(train_data))
	print('train_labels  type:',  type(train_labels))
	print('val_data  type:',  type(val_data))
	print('val_labels  type:', type(val_labels))


	print('\ntrain_data length', len(train_data))
	print('train_labels length', len(train_labels))
	print('val_data length:', len(val_data))
	print('val_labels length:', len(val_labels))

	# Convert to numpy array since TF likes those better than lists
	train_data = np.asarray(train_data,dtype=np.float64)
	train_labels = np.asarray(train_labels)
	val_data = np.asarray(val_data,dtype=np.float64)
	val_labels = np.asarray(val_labels)

	print('\ntrain_data type:',  type(train_data))
	print('train_labels  type:',  type(train_labels))
	print('val_data  type:',  type(val_data))
	print('val_labels  type:', type(val_labels))

	print('\ntrain_data.shape:', train_data.shape)
	print('train_labels.shape:', train_labels.shape)
	print('val_data.shape:', val_data.shape)
	print('val_labels.shape:', val_labels.shape)
	
	# reshape train and val data to 2D for 2D CNN
	dim = train_data.shape
	train_data = train_data.reshape(dim[0],dim[1],dim[2],1)
	dim = val_data.shape
	val_data = val_data.reshape(dim[0],dim[1],dim[2],1)

	print('\ntrain_data.shape:', train_data.shape)
	print('val_data.shape:', val_data.shape)

	return(train_data,train_labels,val_data,val_labels)

def load_tf(train_data,train_labels,val_data,val_labels):
	'''
	Input Arguements:
	train_data = numpy array of training data
	train_labels = numpy array of training labels
	val_data = numpy array of val data
	val_labels = numpy array of val labels

	returns tf datasets for use directly in training
	
	***
	In future this function may merge with load_data
		- Kept separate for now incase there is different preprocessing people want
	***
	'''

	train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
	val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))

	BATCH_SIZE = 64 # Training batch size 
	SHUFFLE_BUFFER_SIZE = 512 # Determines randomness? Need to doublecheck

	# Create training and val batches

	train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
	#val_dataset = val_dataset.batch(BATCH_SIZE)
	val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

	print('\ntrain_dataset type: ', type(train_dataset))
	print('val_dataset type: ', type(val_dataset))
	return train_dataset,val_dataset

'''
Testing with the Code below
'''

if __name__=='__main__':
	train_data,train_labels,val_data,val_labels = load_data('phone_accel_dir')
	train_dataset,val_dataset=load_tf(train_data,train_labels,val_data,val_labels)
