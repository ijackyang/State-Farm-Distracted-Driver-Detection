import numpy as np
from keras import layers
from keras.layers import Input, Add, Lambda,Dense,Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
import keras.backend as K
from image_process import *
from keras.callbacks import Callback
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.optimizers import Adam
from soft_label import soft_label_filename_fetch
import sys
from imgaug import augmenters as iaa
import imgaug as ia
from image_augmentation import image_aug

K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def L2Norm(x):
	return K.l2_normalize(x,axis=-1)


def VGG():
        base_model = VGG16(weights='imagenet')
        base_model.layers.pop()
	X = base_model.get_layer('fc2').output
	X = Dense(10,activation='softmax')(X)
	model = Model(inputs=base_model.input,outputs=X)
        model.summary()
        return model

def Resnet50():
    	"""
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
        
        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes
        
        Returns:
        model -- a Model() instance in Keras
        """
        base_model = ResNet50(weights='imagenet')
        X = base_model.get_layer('flatten_1').output
	X = Dense(10,activation='softmax')(X)
        model = Model(inputs=base_model.input,outputs=X)
        #for layer in base_model.layers[:-14]:
        #        layer.trainable = False
	#model.summary()
        return model


def Resnet50_DO(fixed_layer = -14):
    	"""
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
        
        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes
        
        Returns:
        model -- a Model() instance in Keras
        """
        base_model = ResNet50(weights='imagenet')
        X = base_model.get_layer('flatten_1').output
	X = Dropout(0.5)(X)
	X = Dense(128,activation='relu')(X)
	X = Dense(10,activation='softmax')(X)
        model = Model(inputs=base_model.input,outputs=X)
        for layer in base_model.layers[:fixed_layer]:
                layer.trainable = False
	model.summary()
        return model

class Check_and_Save(Callback):
     def __init__(self, model,X_test,Y_test):
           self.model = model
           self.X_test = X_test
           self.Y_test = Y_test
     def on_epoch_end(self, epoch, logs={}):
           preds = self.model.evaluate(self.X_test,self.Y_test)
           print ("Current Epoch Test Loss = " + str(preds[0]))
           print ("Current Epoch Test Accuracy = " + str(preds[1]))

def shuffle_train(X,Y):
	X = list(X)
	Y = list(Y)
	Z = list(zip(X,Y))
	np.random.shuffle(Z)
	X,Y = zip(*Z)
	return np.asarray(X),np.asarray(Y)


def ResNet50_Train(epochs,mini_batch_size):
    mean_pixel = np.array([103.939, 116.779, 123.68])
    test_name_list = test_filename_fetch()
    prediction = {}
    for i in range(len(test_name_list)):
	prediction[os.path.basename(test_name_list[i])] = np.zeros(10)
    model = Resnet50()
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) 
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.save_weights('initial_weights.h5')
    for i in range(1,5):
	 print ("Here is Cross Validation Round : "+str(i))
	 model.load_weights('initial_weights.h5')
	 train_name_list,train_label_list,dev_name_list,dev_label_list = cross_validation(i)
	 train_batch_size = len(train_name_list)/8
	 dev_batch_size = len(dev_name_list)/8
	 X_dev,Y_dev    = dev_dataset_fetch(dev_name_list,dev_label_list)
	 X_dev = (X_dev - mean_pixel)/255.
	 for k in range(epochs):
		#model.save_weights('temporary_weights.h5')
		#if k==0:
		#	for layer in model.layers[:-2]:
		#		layer.trainable = False
    		#	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
		#	model.summary()
		#else:
		#	for layer in model.layers:
		#		layer.trainable = True
    		#	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
		#	model.summary()
		#model.load_weights('temporary_weights.h5')
		print ("epochs="+str(k))
         	for j in range(8):
            		X_train,Y_train = dataset_fetch(train_name_list,train_label_list,j*train_batch_size,(j+1)*train_batch_size)
	 		#X_dev,Y_dev     = dataset_fetch(dev_name_list,dev_label_list,j*dev_batch_size,(j+1)*dev_batch_size)
			#X_dev = (X_dev - mean_pixel)/255.
            		print('batch_num ='+str(j)+'/8')
            		print ("number of training examples = " + str(X_train.shape[0]))
            		print ("number of dev examples = " + str(X_dev.shape[0]))
            		print ("X_train shape: " + str(X_train.shape))
            		print ("Y_train shape: " + str(Y_train.shape))
            		print ("X_dev shape: " + str(X_dev.shape))
            		print ("Y_dev shape: " + str(Y_dev.shape))
            		check_and_save = Check_and_Save(model,X_dev,Y_dev)
			X_train_t = (X_train - mean_pixel)/255.
            		model.fit(X_train_t, Y_train, epochs = 1, batch_size = mini_batch_size,callbacks=[check_and_save])
			for kk in range(4):
				X_train,Y_train = shuffle_train(X_train,Y_train)
				X_train_t = image_aug(X_train)
				X_train_t = (X_train_t - mean_pixel)/255.
            			model.fit(X_train_t, Y_train, epochs = 1, batch_size = mini_batch_size,callbacks=[check_and_save])
		weights_name = "res50_cross_"+str(i)+"epochs_"+str(k)+".h5"
		model.save_weights(weights_name)
	 test_name_list = test_filename_fetch()
	 test_batch_size = 5120
	 number_batch = len(test_name_list)/test_batch_size
         for m in range(number_batch):
        	 print ("Here is prediction part round :" + str(m))
       		 X_test,name_test = test_dataset_fetch(test_name_list,m*test_batch_size,(m+1)*test_batch_size)
       		 Y_test = model.predict(X_test)
       		 for n in range(X_test.shape[0]):
               		 prediction[name_test[n]] = Y_test[n]
   	 print ("Here is prediction part last round")
   	 X_test,name_test = test_dataset_fetch(test_name_list,number_batch*test_batch_size,len(test_name_list))
   	 Y_test = model.predict(X_test)
   	 for n in range(X_test.shape[0]):
       		 prediction[name_test[n]] = Y_test[n]
	 submit_file = 'submit_cross_'+str(i)+'.csv'
    	 submission_file = open(submit_file,'aw')
    	 with submission_file:
		csv.writer(submission_file).writerow(['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
		for key,value in prediction.iteritems():
			row = [key] + value.tolist()
			csv.writer(submission_file).writerow(row)


def ResNet50_Soft_Train(epochs,mini_batch_size):
    mean_pixel = np.array([103.939, 116.779, 123.68])
    test_name_list = test_filename_fetch()
    prediction = {}
    for i in range(len(test_name_list)):
	prediction[os.path.basename(test_name_list[i])] = np.zeros(10)
    model = Resnet50()
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) 
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    chosen_soft_size = 10240
    basepath = './'
    result_path =  './submit.csv'
    soft_name_list,soft_label_list = soft_label_filename_fetch(threshold=0,path=result_path)
    for i in range(5):
	 current_weights = basepath+'res50_cross_{}epochs_0.h5'.format(i)
    	 model.load_weights(current_weights)
         train_name_list,train_label_list,dev_name_list,dev_label_list = cross_validation(i)
	 train_name_list_c  =  soft_name_list[i*chosen_soft_size:(i+1)*chosen_soft_size] + train_name_list
	 train_label_list_c =  soft_label_list[i*chosen_soft_size:(i+1)*chosen_soft_size] + train_label_list
	 shuffle_index = range(len(train_name_list_c))
	 train_name_list = []
	 train_label_list = []
	 for tt in range(len(train_name_list_c)):
		train_name_list.append(train_name_list_c[shuffle_index[tt]])
		train_label_list.append(train_label_list_c[shuffle_index[tt]])
	 train_batch_size = len(train_name_list)/10
	 X_dev,Y_dev     = dev_dataset_fetch(dev_name_list,dev_label_list)
	 X_dev = (X_dev - mean_pixel)/255.
	 for k in range(epochs):
		print ("epochs="+str(k))
         	for j in range(10):
            		X_train,Y_train = dataset_fetch(train_name_list,train_label_list,j*train_batch_size,(j+1)*train_batch_size)
			X_train = (X_train - mean_pixel)/255.
            		print("medium batch num ",'j='+str(j)+"/10")
            		print ("number of training examples = " + str(X_train.shape[0]))
            		print ("number of dev examples = " + str(X_dev.shape[0]))
            		print ("X_train shape: " + str(X_train.shape))
            		print ("Y_train shape: " + str(Y_train.shape))
            		print ("X_dev shape: " + str(X_dev.shape))
            		print ("Y_dev shape: " + str(Y_dev.shape))
            		check_and_save = Check_and_Save(model,X_dev,Y_dev)
            		model.fit(X_train, Y_train, epochs = 2, batch_size = mini_batch_size,callbacks=[check_and_save])
		weights_name = "res50_sof_"+str(i)+"_epochs_"+str(k)+".h5"
		model.save_weights(weights_name)
	 test_name_list = test_filename_fetch()
	 test_batch_size = 4096
	 number_batch = len(test_name_list)/test_batch_size
         for m in range(number_batch):
        	 print ("Here is prediction part round :" + str(m))
       		 X_test,name_test = test_dataset_fetch(test_name_list,m*test_batch_size,(m+1)*test_batch_size)
       		 Y_test = model.predict(X_test)
       		 for n in range(X_test.shape[0]):
               		 prediction[name_test[n]] = prediction[name_test[n]] + Y_test[n]
   	 print ("Here is prediction part last round")
   	 X_test,name_test = test_dataset_fetch(test_name_list,number_batch*test_batch_size,len(test_name_list))
   	 Y_test = model.predict(X_test)
   	 for n in range(X_test.shape[0]):
       		 prediction[name_test[n]] = prediction[name_test[n]] + Y_test[n]
	 submit_file = 'submit_soft_'+str(i)+'.csv'
    	 submission_file = open(submit_file,'aw')
    	 with submission_file:
		csv.writer(submission_file).writerow(['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
		for key,value in prediction.iteritems():
			row = [key] + value.tolist()
			csv.writer(submission_file).writerow(row)

if __name__ == '__main__':
    #ResNet50_Train(epochs=1,mini_batch_size=32)
    ResNet50_Soft_Train(epochs=1,mini_batch_size=32)
