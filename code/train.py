from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
#K.set_image_dim_ordering('th')
import numpy as np
import sys
from os.path import isfile, join
import shutil
import h5py
import os.path
import glob
import audio_tools as aud

FRAME_ROWS = 128
FRAME_COLS = 128
NFRAMES = 9
MARGIN = NFRAMES/2
COLORS = 1
CHANNELS = COLORS*NFRAMES
TRAIN_PER = 0.8
LR = 0.01
nb_pool = 2
BATCH_SIZE = 16
DROPOUT = 0.25 
DROPOUT2 = 0.5
EPOCHS = 3
FINETUNE_EPOCHS = 10
activation_func2 = 'tanh'

respath = 'results/'
weight_path = join(respath,'weights/')
alldatapath = 'dataset/data/'

def load_data(datapath, part):
	viddata_path = join(datapath,'viddata{}.npy'.format(part))
	auddata_path = join(datapath,'auddata{}.npy'.format(part))
	if isfile(viddata_path) and isfile(auddata_path):
		print ('Loading data...')
		viddata = np.load(viddata_path)
		auddata = np.load(auddata_path)
	#	vidctr = len(auddata)
		print ('Done.')
		return viddata, auddata
	else:
		print ('Preprocessed data not found.')
		sys.exit()
'''
def split_data(viddata, auddata):
	vidctr = len(auddata)
	Xtr = viddata[:int(vidctr*TRAIN_PER),:,:,:]
	Ytr = auddata[:int(vidctr*TRAIN_PER),:]
	Xte = viddata[int(vidctr*TRAIN_PER):,:,:,:]
	Yte = auddata[int(vidctr*TRAIN_PER):,:]
	return (Xtr, Ytr)
'''

def build_model(net_out):
	model = Sequential()
	model.add(Convolution2D(32, 3, padding='same', kernel_initializer='he_normal', data_format='channels_first', input_shape=(CHANNELS, FRAME_ROWS, FRAME_COLS)))
	model.add(BatchNormalization())
	model.add(LeakyReLU())	
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), data_format='channels_first'))
	model.add(Convolution2D(32, 3, padding='same', kernel_initializer='he_normal', data_format='channels_first'))
	model.add(BatchNormalization())
	model.add(LeakyReLU())	
	model.add(Convolution2D(32, 3, padding='same', kernel_initializer='he_normal', data_format='channels_first'))
	model.add(BatchNormalization())
	model.add(LeakyReLU())	
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), data_format='channels_first'))  #Aathira : added padding in all MaxPool
	model.add(Dropout(DROPOUT))
	model.add(Convolution2D(64, 3, padding='same', kernel_initializer='he_normal', data_format='channels_first'))
	model.add(BatchNormalization())
	model.add(LeakyReLU())	
	model.add(Convolution2D(64, 3, padding='same', kernel_initializer='he_normal', data_format='channels_first'))
	model.add(BatchNormalization())
	model.add(LeakyReLU())	
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), data_format='channels_first'))
	model.add(Dropout(DROPOUT))
	model.add(Convolution2D(128, 3, padding='same', kernel_initializer='he_normal', data_format='channels_first'))
	model.add(BatchNormalization())
	model.add(LeakyReLU())	
	model.add(Convolution2D(128, 3, padding='same', kernel_initializer='he_normal', data_format='channels_first'))
	model.add(BatchNormalization())
	model.add(LeakyReLU())	
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), data_format='channels_first'))
	model.add(Dropout(DROPOUT))
	model.add(Convolution2D(128, 3, padding='same', kernel_initializer='he_normal', data_format='channels_first'))
	model.add(BatchNormalization())
	model.add(LeakyReLU())	
	model.add(Convolution2D(128, 3, padding='same', kernel_initializer='he_normal', data_format='channels_first'))
	model.add(BatchNormalization())
	model.add(Activation(activation_func2))	
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), data_format='channels_first'))
	model.add(Dropout(DROPOUT2))
	model.add(Flatten())
	model.add(Dense(512, kernel_initializer='he_normal'))
	model.add(BatchNormalization())
	model.add(Activation(activation_func2))	
	model.add(Dropout(DROPOUT2))
	model.add(Dense(512, kernel_initializer='he_normal'))
	model.add(BatchNormalization())
	model.add(Dense(net_out))
	print(model.summary())
	return model
'''
def savedata(Ytr, Ytr_pred, Yte, Yte_pred, respath=respath):
	np.save(join(respath,'Ytr.npy'),Ytr)
	np.save(join(respath,'Ytr_pred.npy'),Ytr_pred)
#	np.save(join(respath,'Yte.npy'),Yte)
#	np.save(join(respath,'Yte_pred.npy'),Yte_pred)
'''
def standardize_data(Xtr, Ytr):
	Xtr = Xtr.astype('float32')
#	Xte = Xte.astype('float32')
	Xtr /= 255
#	Xte /= 255
	xtrain_mean = np.mean(Xtr)
	Xtr = Xtr-xtrain_mean
#	Xte = Xte-xtrain_mean
	Y_means = np.mean(Ytr,axis=0) 
	Y_stds = np.std(Ytr, axis=0)
	Ytr_norm = ((Ytr-Y_means)/Y_stds)
#	Yte_norm = ((Yte-Y_means)/Y_stds)
	return Xtr, Ytr_norm

def train_net(model, Xtr, Ytr_norm, Xvd, Yvd_norm, batch_size=BATCH_SIZE, epochs=EPOCHS, finetune=False, loadexiting=False,epochno=1,datapart=0):
	if loadexiting:
		#newest = max(glob.iglob(weight_path+'*.hdf5'), key=os.path.getctime)
		if(datapart==0):
			newest = weight_path+'weights.{}-{}.hdf5'.format(epochno-1,3)
		else:
			newest = weight_path+'weights.{}-{}.hdf5'.format(epochno,datapart-1)
		print("NEWEST MODEL : "+newest)
		model.load_weights(newest)
		print("Existing model loaded")
	if finetune:
		newest = max(glob.iglob(weight_path+'*.hdf5'), key=os.path.getctime)
		model.load_weights(newest)
		lr = LR/10
	else:
		lr = LR

	adam = Adam(lr=lr)
	model.compile(loss='mean_squared_error', optimizer=adam)
	checkpointer = ModelCheckpoint(filepath=weight_path+'weights.{}-{}.hdf5'.format(epochno,datapart),
		monitor='val_loss', verbose=1, save_best_only=True)
	history = model.fit(Xtr, Ytr_norm, batch_size=batch_size, epochs=epochs,
		verbose=2, validation_data=(Xvd, Yvd_norm), callbacks=[checkpointer])
	#newest = max(glob.iglob(weight_path+'*.hdf5'), key=os.path.getctime)
	#model.load_weights(newest)
	return model


def predict(model, X, Y_means, Y_stds, batch_size=BATCH_SIZE):
	Y_pred = model.predict(X, batch_size=batch_size, verbose=1)
#	Y_pred = (Y_pred*Y_stds+Y_means)
	return Y_pred

def main():
	if not os.path.exists(weight_path):
		os.makedirs(weight_path)
	Xtr=[]
	Ytr=[]
	
	for i in range(5):
		viddata, auddata = load_data(alldatapath,i)	
#	(Xtr,Ytr), (Xte, Yte) = split_data(viddata, auddata)
		Xtr.append(viddata)
		Ytr.append(auddata)
		net_out = Ytr[i].shape[1]
		print(net_out)
		Xtr[i], Ytr[i] = standardize_data(Xtr[i], Ytr[i])
	model = build_model(net_out)	
	model = train_net(model, Xtr[0], Ytr[0], Xtr[-1], Ytr[-1], epochs=1)

#	model = train_net(model, Xtr, Ytr_norm, Xte, Yte_norm, epochs=FINETUNE_EPOCHS, finetune=True)
#	Ytr_pred = predict(model, Xtr, Y_means, Y_stds)
#	Yte_pred = predict(model, Xte, Y_means, Y_stds)
#	savedata(Ytr, Ytr_pred)
	datapart = 1

	for i in range(1,EPOCHS+1):
		print("Epoch no. : ",i)
		while(datapart<4):
			model = train_net(model, Xtr[datapart], Ytr[datapart], Xtr[-1], Ytr[-1], epochs=1,loadexiting=True,epochno=i,datapart=datapart)
			datapart +=1
		datapart = 0

if __name__ == "__main__":
	main()
