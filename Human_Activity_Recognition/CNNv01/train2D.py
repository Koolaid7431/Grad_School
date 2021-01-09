'''
Human Activity Recognition
Load data & Train model
MRH, July 11, 2020
biomedic.ai
'''

import load_data as ldfile
import create_model as cmodel
import tensorflow as tf
from time import time
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
tf.random.set_seed(1234)

def main():
	phone_accel_dir = '/dataset/HAR/wisdm-dataset/raw/phone/accel'
	phone_gyro_dir = '/dataset/HAR/wisdm-dataset/raw/phone/gyro'

	watch_accel_dir = '/dataset/HAR/wisdm-dataset/raw/watch/accel'
	watch_gyro_dir = '/dataset/HAR/wisdm-dataset/raw/watch/gyro'

	input_directory = phone_accel_dir  # Dataset on local computer 
	t1 = time()
	train_data,train_labels,val_data,val_labels = ldfile.load_data2D(input_directory) 
	train_dataset,val_dataset = ldfile.load_tf(train_data,train_labels,val_data,val_labels)

	model = cmodel.create_model2D() # Change model type if necessary
	a,l,va,vl = cmodel.train_model(model,train_dataset,val_dataset,model_name="Activity_Model_v01",show_stats = False) # Change name or show plots directly if desired
	t2 = time()
	print('Training DeepCNN in {} seconds.'.format(round(t2-t1)))

if __name__ == '__main__':
	main()