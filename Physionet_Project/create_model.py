'''
PhysioNet 2020 Challenge

Generates model architecture and sets training parameters
Calvin April 19, 2020
Edited by MRH July 30, 2020;  

Notes (MRH):
I used the 2D model from Calvin and then changed it to the model with 7x1 and 3x1 Conv2D
I changed the older model name to create_model2D_OLD to have a track of changes


'''

import os
import numpy as np
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt

@tf.function
def macro_soft_f1(y, y_hat):
	"""Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
	Use probability values instead of binary predictions.

	Args:
		y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
		y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

	Returns:
		cost (scalar Tensor): value of the cost function for the batch

		# MIT License
		#
		# Copyright (c) 2019 Mohamed-Achref MAIZA
		# IGNORE_COPYRIGHT: cleared by OSS licensing
		#
		# Permission is hereby granted, free of charge, to any person obtaining a
		# copy of this software and associated documentation files (the "Software"),
		# to deal in the Software without restriction, including without limitation
		# the rights to use, copy, modify, merge, publish, distribute, sublicense,
		# and/or sell copies of the Software, and to permit persons to whom the
		# Software is furnished to do so, subject to the following conditions:
		#
		# The above copyright notice and this permission notice shall be included in
		# all copies or substantial portions of the Software.
		#
		# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
		# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
		# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
		# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
		# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
		# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
		# DEALINGS IN THE SOFTWARE.
"""
	y = tf.cast(y, tf.float32)
	y_hat = tf.cast(y_hat, tf.float32)
	tp = tf.reduce_sum(y_hat * y, axis=0)
	fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
	fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
	soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
	cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
	macro_cost = tf.reduce_mean(cost) # average on all labels
	return macro_cost

@tf.function
def macro_f1(y, y_hat, thresh=0.5):
	"""Compute the macro F1-score on a batch of observations (average F1 across labels)

	Args:
		y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
		y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
		thresh: probability value above which we predict positive

	Returns:
		macro_f1 (scalar Tensor): value of macro F1 for the batch
		# MIT License
		#
		# Copyright (c) 2019 Mohamed-Achref MAIZA
		# IGNORE_COPYRIGHT: cleared by OSS licensing
		#
		# Permission is hereby granted, free of charge, to any person obtaining a
		# copy of this software and associated documentation files (the "Software"),
		# to deal in the Software without restriction, including without limitation
		# the rights to use, copy, modify, merge, publish, distribute, sublicense,
		# and/or sell copies of the Software, and to permit persons to whom the
		# Software is furnished to do so, subject to the following conditions:
		#
		# The above copyright notice and this permission notice shall be included in
		# all copies or substantial portions of the Software.
		#
		# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
		# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
		# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
		# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
		# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
		# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
		# DEALINGS IN THE SOFTWARE.
"""
	y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
	tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
	fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
	fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
	f1 = 2*tp / (2*tp + fn + fp + 1e-16)
	macro_f1 = tf.reduce_mean(f1)
	return macro_f1



def create_model2D_bce():
	'''
	No inputs. Create a CNN model based on the binary output (a list of nine 0/1 for the labels). 
	This is because of multilabel input
	returns a 2D CNN model architecture
	***
	Assumptions:
		- input as an image of size 12x2500x1
		- After first Convolution, the size will be 6x2500x1
		- After first MaxPooling, the size will be 3x2500x1
		- After Second Convolution, the size will be 1x2500x1
		- There is no possibility to apply MaxPooling for size 1x2500x1
		- Note1: I beleive we don't need MaxPooling at all. I will chack for the next week
		- Note2: The second dimention of filter is 1 for both convolution layers to combine only 12 samples at the same time series

	MRH, July 30, 2020

	***
	'''
	LR = 0.0001
	model = tf.keras.Sequential([
		tf.keras.layers.Conv2D(16,(1,31),activation='relu',input_shape=(35,2500,1)),
		#tf.keras.layers.Conv2D(16,(1,31),activation='relu',input_shape=(13,2500,1)),
		#tf.keras.layers.Conv2D(16,(1,31),activation='relu',input_shape=(15,2500,1)),
		#tf.keras.layers.Conv2D(16,(1,11),activation='relu'),
		tf.keras.layers.Conv2D(32,(5,1),strides=(5,1),activation='relu'),
		#tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Conv2D(64,(7,1),activation='relu'),

		tf.keras.layers.Flatten(),
		#tf.keras.layers.Dense(512, activation='relu'),
		tf.keras.layers.Dense(256, activation='relu'),
		tf.keras.layers.Dropout(0.3),
		tf.keras.layers.Dense(128, activation='relu'),
		#tf.keras.layers.Dense(9),
		tf.keras.layers.Dense(9, activation='sigmoid', name='output'),
		#tf.keras.layers.Dense(9, activation='relu', name='output'),
		])

	# model.compile(optimizer='adam',
	# 	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	# 	metrics=['accuracy'])

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
		loss=tf.keras.metrics.binary_crossentropy,
		metrics=[macro_f1])
	return model

def create_model2D_bin():
	'''
	No inputs. Create a CNN model based on the binary output (a list of nine 0/1 for the labels). 
	This is because of multi-label input
	returns a 2D CNN model architecture
	***
	Assumptions:
		- input as an image of size 13x2500x1
		- After first Convolution (1,31), the size will be 13x2500x1
		- After Second Convolution, the size will be 13x2470x1
		- There is no possibility to apply MaxPooling for size ?x2500x1
		- Note1: I beleive we don't need MaxPooling at all. 
		- Note2: The second dimention of filter is 1 for 2,3,4,and 5th convolution layers to combine samples at the same time series

	MRH, July 30, 2020
	? Modified: Sept 16, 2020

	***
	'''
	LR = 0.0001
	model = tf.keras.Sequential([
		tf.keras.layers.Conv2D(16,(1,31),activation='relu',input_shape=(35,2500,1)),
		#tf.keras.layers.Conv2D(16,(1,31),activation='relu',input_shape=(13,2500,1)),
		#tf.keras.layers.Conv2D(16,(1,31),activation='relu',input_shape=(15,2500,1)),
		#tf.keras.layers.Conv2D(16,(1,11),activation='relu'),
		tf.keras.layers.Conv2D(32,(5,1),strides=(5,1),activation='relu'),
		#tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Conv2D(64,(7,1),activation='relu'),

		tf.keras.layers.Flatten(),
		#tf.keras.layers.Dense(512, activation='relu'),
		tf.keras.layers.Dense(256, activation='relu'),
		tf.keras.layers.Dropout(0.3),
		tf.keras.layers.Dense(128, activation='relu'),
		#tf.keras.layers.Dense(9),
		tf.keras.layers.Dense(9, activation='sigmoid', name='output'),
		#tf.keras.layers.Dense(9, activation='relu', name='output'),
		])

	# model.compile(optimizer='adam',
	# 	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	# 	metrics=['accuracy'])

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
		loss=macro_soft_f1,
		metrics=[macro_f1])
	return model


def train_model(model, train_dataset, val_dataset,model_name = "model", show_stats = False):
	'''
	Input Arguments:
	model = model architecture. Generate through previous functions.
	train_dataset = training dataset, generate with load_data.py
	test_dataset = test dataset, generate with load_data.py
	model_name = name of the model, defult is "model" (Optional)
	show_stats = display plots, default is False, set True to see plots before saving. (Optional)

	returns nothing

	***
	Details:
		- Trains model including validation steps
		- Default of 10 epochs
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
	accuracy=fit.history['macro_f1']
	loss = fit.history['loss']
	val_accuracy = fit.history['val_macro_f1']
	val_loss = fit.history['val_loss']
	epochs = range(len(accuracy))

	# Plot training statistics
	if show_stats:
		# Show plot of training over epochs
		print('Showing plot is ON!')
		plt.figure(figsize=(20, 8))
		plt.clf()
		plt.ylabel("Accuracy")
		plt.xlabel("epoch")
		plt.title('Training and validation Accuracy')
		plt.plot(epochs,accuracy,"b", linewidth=0.8, label="accuracy")
		plt.plot(epochs,val_accuracy,"g", linewidth=0.8, label="validation_accuracy")
		#plt.plot(epochs,loss,"k", linewidth=0.8, label="loss")
		#plt.plot(epochs,val_loss,"r", linewidth=0.8, label="validation_loss")
		plt.legend(loc="lower right")
		# plt.Show()
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

	return  loss, accuracy, val_loss, val_accuracy
