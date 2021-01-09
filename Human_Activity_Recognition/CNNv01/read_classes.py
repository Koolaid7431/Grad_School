import os
# from scipy.io import loadmat



# Find files.
def read_files(input_directory):
	input_files = []
	for f in os.listdir(input_directory):
		if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
			input_files.append(f)

	return input_files

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

def class_distribution(input_directory,files):
	'''This function return the list of signals in each class.
	Input: input_directory: path to the input files
			files: the list of files
	Output: classlist: a dictionary with keys equal to the class-labels and value is a list of signals in each class
	MRH, modified from number of classlist to the list of signals of each class, June 22
	'''
	
	# Dctionary "classes" is used for recording the list of files (signals) for each class
	# classes = {'AF': 0,'I-AVB':0,'LBBB':0,'Normal':0, 'PAC':0,'PVC':0, 'RBBB':0, 'STD':0, 'STE':0}
	classlist = {}
	doubleLabel = {}
	tripleLabel = {}
	multiLabel = {}
	# multiLabel = {'AF': 0,'I-AVB':0,'LBBB':0,'Normal':0, 'PAC':0,'PVC':0, 'RBBB':0, 'STD':0, 'STE':0}
	
	count2 = 0 # Number of signals with TWO classes
	count3 = 0 #Number of signals with more than TWO classes
	
	
	for f in files:
		g = f.replace('.mat','.hea')
		input_file = os.path.join(input_directory,g)
		with open(input_file,'r') as f2:
			for lines in f2:
				if lines.startswith('#Dx'): # The line for class-label starts with #Dx 
					tmp = lines.split(': ')[1].split(',') # if signal labeled with more than one class, it's separated by ','
					tmp[-1] = tmp[-1].strip()
					number_of_classes = len(tmp)
	
					if number_of_classes > 2:
						# print('--> Signal {} contains {} classe: {}'.format(g[0:5],number_of_classes,tmp))
						count3 += 1  # counting signals which were labeled with more than TWO class                  
						key = tmp[0] + '+' + tmp[1] + '+' + tmp[2]
						if key in tripleLabel.keys():
							tripleLabel[key].append(f)
						else:
							tripleLabel[key] = [f]
						for key in tmp:
							if key in classlist.keys():
								classlist[key].append(f)
							else:
								classlist[key] = [f]

							if key in multiLabel.keys():
								multiLabel[key] += 1
							else:
								multiLabel[key] = 1


					elif number_of_classes > 1:
						# print('Signal {} contains {} classe: {}'.format(g[0:5],number_of_classes,tmp))
						count2 += 1  # counting signals which were labeled with TWO class 
						key = tmp[0] + '+' + tmp[1]
						if key in doubleLabel.keys():
							doubleLabel[key].append(f)
						else:
							doubleLabel[key] = [f]
						for key in tmp:
							if key in classlist.keys():
								classlist[key].append(f)
							else:
								classlist[key] = [f]

							if key in multiLabel.keys():
								multiLabel[key] += 1
							else:
								multiLabel[key] = 1

					else:
						key = tmp[0]
						if key in classlist.keys():
							classlist[key].append(f)
						else:
							classlist[key] = [f]

	print('\nNumber of signals with TWO classes: ',count2)   
	# print('\nDistribution of signals with TWO classes: \n')
	# for key in doubleLabel.keys():
	#	print(key,'  ',len(doubleLabel[key]))
		
	print('\nNumber of signals with more than two classes: ',count3) 
	# print('\nDistribution of signals with THREE classes: \n')
	# for key in tripleLabel.keys():
	#	print(key,'  ',len(tripleLabel[key]))
	
	
	return classlist, doubleLabel, tripleLabel, multiLabel  