'''
PhysioNet 2020 Challenge

This code finds the best window (5-second) in the signal to be used for training and later for the classifier
This code uses only lead #2 (data[1][indx:indx+2500]) of 12-lead signal
MRH May 21, 2020; 

This code was updated for data augmentation. It will select more windows from the signals with lower occurance in dataset. The new 
function named select_windows is for this new selection.
July 14, 2020 

inputs:
	- input_directory: the directory containing the files
	- file_name
	- win_size: with default set to 5 second
output:
	- The address (index of signal array) for starting the window (one interger number)
Functions:
	- butter_bandpass: for bandpass filtering, used in butter_bandpass_forward_backward_filter
	- butter_bandpass_forward_backward_filter: for forward_backward filtering

	- *** NEEDS TO IMPORT Detectors, panPeakDetect, searchBack functions from ecg_detectors.ecgdetectors 
	- Note: These functions are borrowed from the github account: https://github.com/marianpetruk/ECG_analysis
To see the selected window for a signal:
	uncomment the line before the last line :)
	# plot1v(data[1],[start_window, start_window + win_size * fs],fs)
Future work: 
	- Trying differnt statistics and rules for selecting window
Comments added: June 12, 2020
'''

import os
import numpy as np
import scipy.io as sio
from scipy.io import loadmat
from scipy.signal import butter, sosfilt, sosfiltfilt
from ecg_detectors.ecgdetectors import Detectors, panPeakDetect, searchBack
import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	sos = butter(order, [low, high], analog=False, btype="band", output="sos")
	return sos

def butter_bandpass_forward_backward_filter(data, lowcut, highcut, fs, order=5):
	sos = butter_bandpass(lowcut, highcut, fs, order=order)
	y = sosfiltfilt(sos,
                    data)  # Apply a digital filter forward and backward to a signal.This function applies a linear digital filter twice, once forward and once backwards. The combined filter has zero phase and a filter order twice that of the original.
	return y

def plot1v(x,p,fs): 
# x and y are from different signals and p shows the position btw signals
	times = np.arange(x.shape[0], dtype='float') / fs
	print('Length of x = {}, Length of times = {}'.format(len(x),len(times)))
	plt.figure(figsize=(20, 8))
	plt.clf()

	plt.ylabel("Amplitude (dB)")
	plt.xlabel("Time (s)")
	plt.plot(times, x, "g", linewidth=1, label="Original signal")
	plt.legend(loc="lower left")
	plt.twinx()

	ymin = np.min(x)
	ymax = np.max(x)
	alpha = 0.2 * (ymax - ymin)
	ymax += alpha
	ymin -= alpha

	p = np.asarray(p,dtype=np.float64)
	plt.vlines(p/fs, ymin, ymax,
           color="r",
           linewidth=0.7,
           label="window selected")
	plt.grid(True)
	plt.axis("tight")
	plt.legend(loc="upper right")
	plt.show()

def pan_tompkins_detector(raw_ecg, mwa, fs, N):
	N = int(N / 100 * fs)
	mwa_peaks = panPeakDetect(mwa, fs)
	r_peaks = searchBack(mwa_peaks, raw_ecg, N)
	return r_peaks

def load_challenge_data(filename): # This function was borrowed from PhysioNet challenge
	x = loadmat(filename)
	data = np.asarray(x['val'], dtype=np.float64)

	new_file = filename.replace('.mat','.hea')
	input_header_file = os.path.join(new_file)

	with open(input_header_file,'r') as f:
		header_data=f.readlines()

	return data, header_data


def select_win(input_directory, file_name, win_size=5):
	fs = 500 # sampling ferequency
	window_size = win_size * fs # 5 second
	step_size = fs # step for sliding window. if equal to fs, means 1 second
	lowcut = 0.05 * 3.3  # 9.9 beats per min
	highcut = 15  # 900 beats per min
	integration_window = 50  # Change proportionally when adjusting frequency (in samples)

	list_of_features = []
	filename = os.path.join(input_directory, file_name)
	x = sio.loadmat(filename)
	data = np.asarray(x['val'], dtype=np.float64)
	data_len = len(data[1])

	# print('------------------------------------- File-name: {}'.format(file_name.split('.')[0]))
	# print('File length is: {}'.format(len(data[1])))

	# Slice data into win_size (second) windows
	indx = 0 # used to slice signal in 2500 segments
	counter = 0 # counting number of 5-sec windows for each file

	# The features dictionary saves several statistics for all 5-sec segments of the signal, but (for now) onlt std of hear-rate is used
	# to select the best 5-sec window. Also R-peaks are used to make sure that 5-sec segment starts from one R-Peak 
	features = {'file_name':file_name.split('.')[0],'length':data_len,'num_of_win':0,'mean_HR':[],'std_HR':[],
			'min_HR':[],'max_HR':[],'R_Peaks':[]} 

	while indx <=(data_len - window_size):
		#print('index = ', indx)
		window = data[1][indx:indx+2500]
		indx += step_size # step for window-sliding is equal to one second
		filtered_signal = butter_bandpass_forward_backward_filter(window, lowcut, highcut, fs, order=4)
		# Derivative - provides QRS slope information.
		differentiated_ecg_measurements = np.ediff1d(filtered_signal)
		# Squaring - intensifies values received in derivative. 
		# This helps restrict false positives caused by T waves with higher than usual spectral energies..
		squared_ecg_measurements = differentiated_ecg_measurements ** 2
		# Moving-window integration.
		integrated_ecg_measurements = np.convolve(squared_ecg_measurements, np.ones(integration_window))
		# Fiducial mark - peak detection on integrated measurements.
		rpeaks = pan_tompkins_detector(window, integrated_ecg_measurements, fs, integration_window)
		# to remove duplicate R-peaks
		rpeaks = list(dict.fromkeys(rpeaks))
		# print('R-peaks positions are: ',rpeaks)
		if len(rpeaks) < 2: # for the case that less than 2 R-Peaks was found
			rpeaks = [0,2500]

		rr = np.diff(rpeaks) / fs * 1000  # RR-interval in miliseconds
		hr = 60 * 1000 / rr  # Heart-rate per minute
		# print("RR-interval in miliseconds =", rr)
		# np.set_printoptions(precision=0)
		# print("Heart-rate per minute =", hr)

		# Mean HR
		mean_hr = np.mean(hr)

		# STD HR
		std_hr = np.std(hr)
		# print("\nMean HR =", round(mean_hr, 3), "±", round(std_hr, 3))
		# print("std_hr =", round(std_hr, 3))

		# Min HR
		min_hr = np.min(hr)
		# print("min_HR =", round(min_hr, 3))

		# Max HR
		max_hr = np.max(hr)
		# print("max_HR =", round(max_hr, 3), "\n")

		features['mean_HR'].append(round(mean_hr,1))
		features['std_HR'].append(round(std_hr,1))		
		features['min_HR'].append(round(min_hr,1))
		features['max_HR'].append(round(max_hr,1))
		features['R_Peaks'].append(rpeaks)
		counter += 1

	# print('Number of windows in this file is: {}'.format(counter))
	features['num_of_win'] = counter
	std_list = features['std_HR']
	min_max_dif_list = np.asarray(features['max_HR'], dtype= np.float64) - np.asarray(features['min_HR'], dtype=np.float64)

	# print('list of std of mean-HR:')
	# print(std_list)
	# print('list of different btw min & max:')
	# print(min_max_dif_list)

	win_num = np.argmax(std_list)
	# print('Win-Number (from 0) = ',win_num)
	# print('R_Peaks = ', features['R_Peaks'][win_num])

	# The rest of the code is to find the first R-Peak in the window
	# There are some treshold for finding the actual R-Peaks (from some peaks that are similar to R-Peaks) 
	# The treshold was chosen based on files that code returns not-good start points. This treshold works for almost all signals
	start_window = win_num * step_size
	first_peak_location = features['R_Peaks'][win_num][0]
	second_peak_location = features['R_Peaks'][win_num][1]
	if (first_peak_location < 200) or (second_peak_location-first_peak_location <200): 
		first_peak_location = second_peak_location
		# print('Second peak selected (first selected window)!')
	if len(features['R_Peaks'][win_num])>2: # if more than two R-Peaks were found, we check the third peak for better R-Peak
		third_peak_location = features['R_Peaks'][win_num][2]
		if (second_peak_location < 200) or (third_peak_location - second_peak_location < 200):
			first_peak_location = third_peak_location
			# print('Third peak selected (first selected window)!')

	if (start_window + first_peak_location) <= (data_len - window_size): # To make sure that the window doesn't pass the signal size
		start_window += first_peak_location
	elif win_num > 0: # If window passed the signal size and there is a window before it, we choose the previous window
		win_num -= 1 
		start_window = win_num * step_size
		first_peak_location = features['R_Peaks'][win_num][0]
		second_peak_location = features['R_Peaks'][win_num][1]
		if (first_peak_location < 200) or (second_peak_location-first_peak_location <200) :
			first_peak_location = second_peak_location
			# print('Second peak selected (one window before)!')
		if len(features['R_Peaks'][win_num])>2:
			third_peak_location = features['R_Peaks'][win_num][2]
			if (second_peak_location < 200) or (third_peak_location - second_peak_location < 200):
				first_peak_location = third_peak_location
				# print('Third peak selected (one window before)! ')
		#start_window += first_peak_location
		if (start_window + first_peak_location) < (data_len - window_size): 
			start_window += first_peak_location
		elif (start_window + features['R_Peaks'][win_num][0]) < (data_len - window_size):
			start_window += features['R_Peaks'][win_num][0]


	# plot1v(data[1],[start_window, start_window + win_size * fs],fs)
	return start_window

def start_peak(win_num,features,step_size,data_len,window_size):
	start_window = win_num * step_size
	first_peak_location = features['R_Peaks'][win_num][0]
	second_peak_location = features['R_Peaks'][win_num][1]
	if (first_peak_location < 200) or (second_peak_location-first_peak_location <200): 
		first_peak_location = second_peak_location
		# print('Second peak selected (first selected window)!')
	if len(features['R_Peaks'][win_num])>2: # if more than two R-Peaks were found, we check the third peak for better R-Peak
		third_peak_location = features['R_Peaks'][win_num][2]
		if (second_peak_location < 200) or (third_peak_location - second_peak_location < 200):
			first_peak_location = third_peak_location
			# print('Third peak selected (first selected window)!')

	if (start_window + first_peak_location) <= (data_len - window_size): # To make sure that the window doesn't pass the signal size
		start_window += first_peak_location
	elif win_num > 0: # If window passed the signal size and there is a window before it, we choose the previous window
		win_num -= 1 
		start_window = win_num * step_size
		first_peak_location = features['R_Peaks'][win_num][0]
		second_peak_location = features['R_Peaks'][win_num][1]
		if (first_peak_location < 200) or (second_peak_location-first_peak_location <200) :
			first_peak_location = second_peak_location
			# print('Second peak selected (one window before)!')
		if len(features['R_Peaks'][win_num])>2:
			third_peak_location = features['R_Peaks'][win_num][2]
			if (second_peak_location < 200) or (third_peak_location - second_peak_location < 200):
				first_peak_location = third_peak_location
				# print('Third peak selected (one window before)! ')
		#start_window += first_peak_location
		if (start_window + first_peak_location) < (data_len - window_size): 
			start_window += first_peak_location
		elif (start_window + features['R_Peaks'][win_num][0]) < (data_len - window_size):
			start_window += features['R_Peaks'][win_num][0]
	
	return start_window	

def select_windows(input_directory, file_name, win_size=5):
	fs = 500 # sampling ferequency
	window_size = win_size * fs # 5 second
	step_size = fs # step for sliding window. if equal to fs, means 1 second
	lowcut = 0.05 * 3.3  # 9.9 beats per min
	highcut = 15  # 900 beats per min
	integration_window = 50  # Change proportionally when adjusting frequency (in samples)

	list_of_features = []
	tmp_input_file = os.path.join(input_directory, file_name)
	data,header_data = load_challenge_data(tmp_input_file)
	data_len = len(data[1])
	labels = header_data[15].split()[1].split(',')
	# print('------------------------------------- File-name: {}'.format(file_name.split('.')[0]))
	# print('File length is: {}'.format(len(data[1])))

	# Slice data into win_size (second) windows
	indx = 0 # used to slice signal in 2500 segments
	counter = 0 # counting number of 5-sec windows for each file

	# The features dictionary saves several statistics for all 5-sec segments of the signal, but (for now) onlt std of hear-rate is used
	# to select the best 5-sec window. Also R-peaks are used to make sure that 5-sec segment starts from one R-Peak 
	features = {'file_name':file_name.split('.')[0],'length':data_len,'num_of_win':0,'mean_HR':[],'std_HR':[],
			'min_HR':[],'max_HR':[],'R_Peaks':[]} 

	while indx <=(data_len - window_size):
		#print('index = ', indx)
		window = data[1][indx:indx+2500]
		indx += step_size # step for window-sliding is equal to one second
		filtered_signal = butter_bandpass_forward_backward_filter(window, lowcut, highcut, fs, order=4)
		# Derivative - provides QRS slope information.
		differentiated_ecg_measurements = np.ediff1d(filtered_signal)
		# Squaring - intensifies values received in derivative. 
		# This helps restrict false positives caused by T waves with higher than usual spectral energies..
		squared_ecg_measurements = differentiated_ecg_measurements ** 2
		# Moving-window integration.
		integrated_ecg_measurements = np.convolve(squared_ecg_measurements, np.ones(integration_window))
		# Fiducial mark - peak detection on integrated measurements.
		rpeaks = pan_tompkins_detector(window, integrated_ecg_measurements, fs, integration_window)
		# to remove duplicate R-peaks
		rpeaks = list(dict.fromkeys(rpeaks))
		# print('R-peaks positions are: ',rpeaks)
		if len(rpeaks) < 2: # for the case that less than 2 R-Peaks was found
			rpeaks = [0,2500]

		rr = np.diff(rpeaks) / fs * 1000  # RR-interval in miliseconds
		hr = 60 * 1000 / rr  # Heart-rate per minute
		# print("RR-interval in miliseconds =", rr)
		# np.set_printoptions(precision=0)
		# print("Heart-rate per minute =", hr)

		# Mean HR
		mean_hr = np.mean(hr)

		# STD HR
		std_hr = np.std(hr)
		# print("\nMean HR =", round(mean_hr, 3), "±", round(std_hr, 3))
		# print("std_hr =", round(std_hr, 3))

		# Min HR
		min_hr = np.min(hr)
		# print("min_HR =", round(min_hr, 3))

		# Max HR
		max_hr = np.max(hr)
		# print("max_HR =", round(max_hr, 3), "\n")

		features['mean_HR'].append(round(mean_hr,1))
		features['std_HR'].append(round(std_hr,1))		
		features['min_HR'].append(round(min_hr,1))
		features['max_HR'].append(round(max_hr,1))
		features['R_Peaks'].append(rpeaks)
		counter += 1

	# print('Number of windows in this file is: {}'.format(counter))
	features['num_of_win'] = counter
	std_list = features['std_HR'] # Choosing standard deviation of heart-rate as a feature for selecting window
	min_max_dif_list = np.asarray(features['max_HR'], dtype= np.float64) - np.asarray(features['min_HR'], dtype=np.float64)

	# print('list of std of mean-HR:')
	# print(std_list)
	# print('list of different btw min & max:')
	# print(min_max_dif_list)

	# The following line will find the 5-sec window with highest standard deviation of heart-rate
	win_num = np.argmax(std_list)
	# print('Win-Number (from 0) = ',win_num)
	# print('R_Peaks = ', features['R_Peaks'][win_num])

	# The rest of the code is to find the first R-Peak in the window
	# There are some treshold for finding the actual R-Peaks (from some peaks that are similar to R-Peaks) 
	# The treshold was chosen based on files that code returns not-good start points. This treshold works for almost all signals
	start_windows = []
	start_win = start_peak(win_num,features,step_size,data_len,window_size)# Find R-peak in given 5-sec window 
	start_windows.append(start_win)
	
	strid = 7
	for label in labels:
		if label in ['164909002','164931005']: # These two classes contain 236 and 220 signals out of 6877 signals
			strid = 1
		elif label in ['164884008','270492004','284470004','426783006','429622005']: # 700 to 900 signals out of 6877
			strid = 3

	
	# print('counter= ', counter)
	# print('data_len= ', data_len)
	# print('labels= ', labels)

	# this part is for data augmentation.It starts from first 5-sec win in the signal and based on the "strid" that was 
	# chosen in above lines, slides on the signal to select more 5-sec windows
	win_num = 0
	while win_num <(counter): 
		# print('win_num = ', win_num)
		start_win = start_peak(win_num,features,step_size,data_len,window_size)
		start_windows.append(start_win)
		win_num += strid # step for window-sliding is equal to one second


	# plot1v(data[1],[start_win, start_win + win_size * fs],fs)
	return start_windows


