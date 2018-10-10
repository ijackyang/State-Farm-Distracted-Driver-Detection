import numpy as np
import csv


def string_to_float(arr):
	result = []
	for i in range(1,len(arr)):
		result.append(float(arr[i]))
	return np.asarray(result)

def average_results(outfile='submit.csv'):
    res_dict = dict()
    path = './submit_soft_4.csv'
    f = open(path,'r')
    line = f.readline()
    while(1):
        line = f.readline()
        if line=='':
                break
        arr = line.strip().split(',')
        res_dict[arr[0]] = string_to_float(arr)
    f.close()
    fileList = []
    for i in range(len(fileList)):
        path = './submit_soft_'+str(fileList[i])+'.csv'
        f = open(path,'r')
        line = f.readline()
        while(1):
                line = f.readline()
                if line=='':
                        break
                arr = line.strip().split(',')
		res_dict[arr[0]] -= string_to_float(arr)
        f.close()
    submission_file = open(outfile,'w')
    with submission_file:
        writer = csv.writer(submission_file)
        writer.writerow(['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
    submission_file = open(outfile,'aw')
    with submission_file:
    	for key,value in res_dict.iteritems():
		value = value/5
		row = [key] + value.tolist()
		csv.writer(submission_file).writerow(row)	
if __name__ == '__main__':
     #average_results(outfile='submit_soft_01.csv')
     average_results(outfile='submit_soft_01234.csv')
