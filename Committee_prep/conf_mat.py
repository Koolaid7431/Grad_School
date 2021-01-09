import os 
import numpy as np
import tensorflow as tf
import load_data as ldfile
from sklearn.preprocessing import MultiLabelBinarizer
from select_win import select_win,select_windows

def save_challenge_predictions(output_directory,filename,scores,labels,classes):
	recording = os.path.splitext(filename)[0]
	new_file = filename.replace('.mat','.csv')
	output_file = os.path.join(output_directory,new_file)

	# Include the filename as the recording number
	recording_string = '#{}'.format(recording)
	class_string = ','.join(classes)
	label_string = ','.join(str(i) for i in labels)
	score_string = ','.join(str(i) for i in scores)

	with open(output_file, 'w') as f:
		f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')

def read_files(input_directory):
	input_files = []
	for f in os.listdir(input_directory):
		if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
			input_files.append(f)

	return input_files

def test_classifier_files_bin(input_directory,TXTfile,output_directory,model,win_size=5, fs=500,thresh=0.5):
	# inputs:
	# fs: signal frequency
	# Find files. List of files for testing is in the TXTfile which was created in load_data2D() version 04
	input_files = []
	with open(TXTfile,'r') as f:
		file_list=f.readlines()
	for line in file_list:
		input_files.append(line.strip())

	all_files = read_files(input_directory)

	classes = ldfile.get_classes(input_directory,all_files)
	num_classes = len(classes)

	if not os.path.isdir(output_directory):
		os.mkdir(output_directory)

	# Load model.    
	print('Loading Deep CNN model...')

#     loaded_model = tf.keras.models.load_model(model_directory)
#     #model2 = tf.keras.Sequential([loaded_model, tf.keras.layers.Softmax()])
	model2 = model
	model2.summary()

	# Iterate over files.
	print('Reading testing files ...')
	num_files = len(input_files)
	print('Number of files (in TXT file):', num_files)
	counter = 0
	count_correct_classification = {}
	label_true = []
	label_pred = []

	all_test_labels =[]
# 	for i, f in enumerate(input_files):
	for i, f in enumerate(all_files):
		tmp_input_file = os.path.join(input_directory, f)
		data, header_data = ldfile.load_challenge_data(tmp_input_file)
		signal_label = header_data[15].split()[1].split(',')      # [0] # Only take first
		all_test_labels.append(signal_label)

	# Fit the multi-label binarizer on the testing set
	print("Labels:")
	mlb = MultiLabelBinarizer()
	mlb.fit(all_test_labels)
	# Loop over all labels and show them
	N_LABELS = len(mlb.classes_)
	for (i, label) in enumerate(mlb.classes_):
		print("{}. {}".format(i, label))

	flag = True
	for i, f in enumerate(input_files):
		tmp_input_file = os.path.join(input_directory, f)
		data, header_data = ldfile.load_challenge_data(tmp_input_file)

		# preparing data for classifier     
		test_data = []
		test_labels = []
		#win_size=5 # 5 second window
		win_start = select_win(input_directory, f, win_size)

		temp = []
		for i in [0,4,10,11,1,2,5,6,7,8,9,3,4]: 
			temp.append(data[i][win_start:win_start+win_size*fs])
		test_data.append(temp) 
		test_data = np.asarray(test_data,dtype=np.float64)
		dim = test_data.shape
		test_data = test_data.reshape(dim[0],dim[1],dim[2],1)

		signal_label = header_data[15].split()[1].split(',')      # [0] # Only take first
		test_labels.append(signal_label)
		# transform the test labels 
		test_labels_bin = mlb.transform(test_labels)
		if flag:
			print('test_labels_bin: ', test_labels_bin)

		# Create batch
		test_data_slice = tf.data.Dataset.from_tensor_slices((test_data, test_labels_bin))
		test_data_slice = test_data_slice.batch(1) # Create batch the same size as the number of samples

		# Applying Model
#         current_label = np.zeros(num_classes, dtype=int)
#         current_score = np.zeros(num_classes)
		y_hat = model2.predict(test_data_slice)[0]
		y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)

		if flag:
			print('y_hat = ', y_hat)
			print('y_pred = ', y_pred)
		if counter > 10:
			flag = False

		# --------------------------------------------------->>> NEW:create list of pred labels
		label_pred.append(y_pred)

		# --------------------------------------------------->>> NEW:create list of true labels
		t_labels = np.zeros(num_classes, dtype=int)
		label_list = header_data[15].split()[1].split(',')
		for tmp_label in label_list: 
			t_labels[classes.index(tmp_label)] = 1        
		label_true.append(t_labels)

		# Save results.        
		save_challenge_predictions(output_directory, f, y_hat, y_pred, classes)

		# Show corrected classified signals
#         if y_pred.sum()>1:
#             print('More than one class was recognized')
		for j in range(len(y_pred)):
			if y_pred[j] == 1:
				tmp_class = classes[j]
				if tmp_class in header_data[15].split()[1].split(','): # if any of labels are detected in multi-labeled signals
					counter += 1
					if tmp_class in count_correct_classification.keys():
						count_correct_classification[tmp_class] += 1
					else:
						count_correct_classification[tmp_class] = 1

	print('Number of correct classification: {}'.format(counter))
	print('percentage of correct classification: {}'.format(counter/len(input_files)))

	return label_true, label_pred, count_correct_classification

def conf_matrix(label_true,label_pred):
	num_classes = len(label_pred[0])
	num_signals = len(label_pred)
	conf_mat = np.zeros((num_classes,num_classes), dtype=np.float64)
	for i in range(num_signals):
		for j in range(num_classes):
			if label_true[i][j] == 1:
				for k in range(num_classes):
					if label_pred[i][k] == 1:
						conf_mat[j][k] += 1
	return conf_mat