'''
PhysioNet 2020 Challenge

Generates model architecture and sets training parameters
Calvin April 19, 2020
Edited by MRH June 5, 2020;  

Notes (MRH):
I used the 2D model from Calvin and then changed it to the model with 7x1 and 3x1 Conv2D
I changed the older model name to create_model2D_OLD to have a track of changes
'''

import os
import numpy as np
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt


def create_model2D():
	'''
	No inputs
	returns a 2D CNN model architecture
	***
	Assumptions:
		- input as an image of size 3x100x1
		- After first Convolution, the size will be 6x2500x1
		- After first MaxPooling, the size will be 3x2500x1
		- After Second Convolution, the size will be 1x2500x1
		- There is no possibility to apply MaxPooling for size 1x2500x1
		- Note1: I beleive we don't need MaxPooling at all. I will chack for the next week
		- Note2: The second dimention of filter-mask is 1 for convolution layers to combine only 3 samples at the same time series

	MRH, July 10, 2020

	***
	'''
	LR = 0.0001 # Learning Rate. Default is 0.001
	model = tf.keras.Sequential([
		tf.keras.layers.Conv2D(16,(1,3),activation='relu',input_shape=(3,100,1)),
		# tf.keras.layers.Conv2D(32,(1,3),activation='relu'),
		tf.keras.layers.Conv2D(64,(3,1),activation='relu'),

		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation='relu'),
		# tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Dense(64, activation='relu'),
		tf.keras.layers.Dense(5),
		])

	#model.compile(optimizer='adam',
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy'])

	return model

def create_model():
	'''
	No inputs
	returns a 1D CNN model architecture
	***
	Assumptions:
		- input size is based on 12 channels 5 seconds long
	This is the primary model used for now
	***
	'''
	model = tf.keras.Sequential([
		tf.keras.layers.Conv1D(16,3,activation='relu',input_shape=(3,100)),
		tf.keras.layers.MaxPooling1D(),
		tf.keras.layers.Conv1D(32,3,activation='relu'),
		tf.keras.layers.MaxPooling1D(),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dense(5),
		])

	model.compile(optimizer='adam',
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy'])

	return model

def train_model(model, train_dataset, val_dataset,model_name = "model", show_stats = False):
	'''
	Input Arguments:
	model = model architecture. Generate through previous functions.
	train_dataset = training dataset, generate with load_data.py
	val_dataset = validation dataset, generate with load_data.py
	model_name = name of the model, defult is "model" (Optional)
	show_stats = display plots, default is False, set True to see plots before saving. (Optional)

	returns nothing

	***
	Details:
		- Trains model including validation steps
		- Default of 20 epochs
		- "Saved_Models" is directory to store output files in
			- will create directory if not present
		- Generates plot of training history and loss for analysis
	***
	'''
	# Train Model
	t1 = time()
	num_of_epochs = 30
	fit = model.fit(train_dataset,validation_data=val_dataset,validation_steps=10,epochs=num_of_epochs)
	t2 = time()
	print('Training time is {} seconds for {} epochs'.format(round(t2-t1),num_of_epochs))
	
	# Save model for future use
	model.save(model_name)
	print('Created directory "{}" to store model.'.format(model_name))

	# Extacting Training information
	accuracy=fit.history['accuracy']
	loss = fit.history['loss']
	val_accuracy = fit.history['val_accuracy']
	val_loss = fit.history['val_loss']
	epochs = range(len(accuracy))

	# Plot training statistics
	if show_stats:
		# Show plot of training over epochs
		print('Showing plot is ON!')
		plt.figure(figsize=(20, 8))
		plt.clf()
		plt.ylabel("Accuracy/loss")
		plt.xlabel("epoch")
		plt.title('Training and validation Accuracy')
		plt.plot(epochs,accuracy,"b", linewidth=0.8, label="accuracy")
		plt.plot(epochs,val_accuracy,"g", linewidth=0.8, label="val_accuracy")
		#plt.plot(epochs,loss,"k", linewidth=0.8, label="loss")
		#plt.plot(epochs,val_loss,"r", linewidth=0.8, label="validation_loss")
		plt.legend(loc="lower right")
		#plt.Show()
		# plt.savefig("accuracy_plots.png")

	else:
		# Automatically save plots
		print('Showing plot is OFF!')
		plt.figure(figsize=(20, 8))
		plt.clf()
		plt.ylabel("Accuracy/loss")
		plt.xlabel("epoch")
		plt.title('Training and validation Accuracy/Loss')
		plt.plot(epochs,accuracy,"b", linewidth=0.8, label="accuracy")
		plt.plot(epochs,val_accuracy,"g", linewidth=0.8, label="validation_accuracy")
		plt.plot(epochs,loss,"k", linewidth=0.8, label="loss")
		plt.plot(epochs,val_loss,"r", linewidth=0.8, label="validation_loss")
		plt.legend(loc="lower right")
		plt.savefig("accuracy_loss.png")

	return accuracy, loss, val_accuracy, val_loss
