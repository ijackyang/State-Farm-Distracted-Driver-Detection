import numpy as np
import csv
from Cross_Validation import string_to_float


def soft_label_filename_fetch(threshold=0.8,path='../result/submit.csv'):
	soft_filename = []
	soft_label    = []
	f = open(path,'r')
	line = f.readline()
	while(1):
		line = f.readline()
		if(line==''):
			break
		arr = line.strip().split(',')
		label = string_to_float(arr)
		if any(label>threshold):
			soft_filename.append('../input/imgs/test/'+arr[0])
			soft_label.append(np.argmax(label))
	f.close()
	return soft_filename,soft_label	
