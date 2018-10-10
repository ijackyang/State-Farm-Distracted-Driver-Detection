import glob
from PIL import Image
import numpy as np
from keras.utils import to_categorical
import csv
import os
from shutil import copy


def dataset_fetch(filenameList,labelList,begin,end):
    image_list = []
    label_list = []
    for j in range(begin,end):
        img = Image.open(filenameList[j])
        img = img.resize((224,224),Image.BILINEAR)
        image_list.append(np.array(img.getdata()).reshape(224,224,3))
        label_list.append(labelList[j])
    #mean_arr  = np.array([103.939, 116.779, 123.68])
    #image_arr = (np.asarraiy(image_list)-mean_arr)/255.
    image_arr = np.asarray(image_list,dtype=np.uint8)
    label_arr = to_categorical(np.asarray(label_list),num_classes=10)
    return image_arr,label_arr

def dev_dataset_fetch(filenameList,labelList):
    image_list = []
    label_list = []
    for j in range(len(filenameList)):
        img = Image.open(filenameList[j])
        img = img.resize((224,224),Image.BILINEAR)
        image_list.append(np.array(img.getdata()).reshape(224,224,3))
        label_list.append(labelList[j])
    image_arr = np.asarray(image_list,dtype=np.uint8)
    label_arr = to_categorical(np.asarray(label_list),num_classes=10)
    return image_arr,label_arr


def driver_id_fetch():
    dr = dict()
    path = os.path.join('..', 'input', 'driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr    


def cross_validation(nfold=0):
	name_list,label_list,unique_driver_list = filename_label_fetch()
	train_name_list_t = []
	train_label_list_t = []
	dev_name_list = []
	dev_label_list = []
	for i in range(nfold*5,(nfold+1)*5):
		dev_name_list  = dev_name_list  + name_list[unique_driver_list[i]]
		dev_label_list = dev_label_list + label_list[unique_driver_list[i]]
	fold_num = len(unique_driver_list)
	for i in range(fold_num):
	    if i<nfold*5 or i>=(nfold+1)*5:
		train_name_list_t = train_name_list_t + name_list[unique_driver_list[i]]
		train_label_list_t = train_label_list_t + label_list[unique_driver_list[i]]
	shuffle_index = range(len(train_name_list_t))
	np.random.shuffle(shuffle_index)
	train_name_list = []
	train_label_list = []
	for i in range(len(train_name_list_t)):
		train_name_list.append(train_name_list_t[shuffle_index[i]])
		train_label_list.append(train_label_list_t[shuffle_index[i]])
	return train_name_list,train_label_list,dev_name_list,dev_label_list

def create_cross_validation_file(nfold=0):
	train_name,train_label,dev_name,dev_label = cross_validation(nfold=nfold)
	main_folder = '../input/nfold_{}'.format(nfold)
	for i in range(len(train_name)):
		destination_folder = main_folder+'/train/c{}'.format(train_label[i])
		copy(train_name[i],destination_folder)
	for i in range(len(dev_name)):
		destination_folder = main_folder+'/dev/c{}'.format(dev_label[i])
		copy(dev_name[i],destination_folder)

def filename_label_fetch():
    label_list = []
    name_list = []
    driver_id_list = []
    driver_id = driver_id_fetch()
    for j in range(10):
        folderName = '../input/imgs/train/c'+str(j)+'/*'
        for filename in glob.glob(folderName):
            name_list.append(filename)
            label_list.append(j)
	    file_name = os.path.basename(filename)
	    file_name = file_name.replace('_mod','')
	    driver_id_list.append(driver_id[file_name])
    total_images = len(name_list)
    shuffle_index = range(total_images)
    np.random.shuffle(shuffle_index)
    np_name_list = []
    np_label_list = []
    np_driver_list = []

    for j in range(total_images):
        np_name_list.append(name_list[shuffle_index[j]])
        np_label_list.append(label_list[shuffle_index[j]])
        np_driver_list.append(driver_id_list[shuffle_index[j]])
    unique_driver_list = sorted(list(set(np_driver_list)))
    name_list = {}
    label_list = {}
    for driver in unique_driver_list:
	name_list[driver] = []
	label_list[driver] = []
    for j in range(total_images):
    	name_list[np_driver_list[j]].append(np_name_list[j])
	label_list[np_driver_list[j]].append(np_label_list[j])

    return name_list,label_list,unique_driver_list


def test_filename_fetch():
	name_list = []
	path = '../input/imgs/test/*'
	for filename in glob.glob(path):
		name_list.append(filename.replace("*","filename"))
	return name_list


def test_dataset_fetch(name_list,begin,end):
    image_list = []
    name_arr   = []
    mean_pixel = np.array([103.939, 116.779, 123.68])
    for j in range(begin,end):
        img = Image.open(name_list[j])
        img = img.resize((224,224),Image.BILINEAR)
        image_list.append(np.array(img.getdata()).reshape(224,224,3))
	name_arr.append(os.path.basename(name_list[j]))
    image_arr = np.asarray(image_list)
    image_arr = (image_arr-mean_pixel)/255.
    name_arr  = np.asarray(name_arr)
    return image_arr,name_arr

