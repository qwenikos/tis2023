
import misc
import matplotlib.pyplot as plt
from keras.layers import Input
from models import create_cnn, create_gated_cnn, create_model
from keras.models import Model
import sys 
import os
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
import random as rn
import numpy as np



np.random.seed(2000)

# Necessary for starting core Python generated random numbers in a well-defined state.
rn.seed(2023)


# applies the grid-search algorithm for hyperparameter tuning in a cnn model
# , and afterward, trains the best cnn model and return it
def train_cnn(train_x, train_y, test_x, test_y, params, params_tuning):

	if params_tuning == 'yes':    

		cnn_model = KerasClassifier(build_fn=create_cnn, verbose=0)

		grid = GridSearchCV(estimator=cnn_model, param_grid=params, n_jobs=-1, cv=5)


		grid_model = grid.fit(train_x, train_y)

		params = grid_model.best_params_


    # mcp = ModelCheckpoint(filepath = tmp_output_directory + "/CNNonRaw.hdf5",
	# 			verbose = 0,
	# 			save_best_only = True)

	earlystopper = EarlyStopping(monitor = 'val_loss', 
					patience = 40,
					min_delta = 0,
					verbose = 1,
					mode = 'auto')

	print ("\t\t\tTraining network.")


	best_cnn = create_cnn(sample_dim=params['sample_dim'], lr=params['lr'], mntm=params['mntm'], flt=params['flt'])
	best_cnn.fit(train_x, train_y, batch_size=params['batch_size'], epochs=params['epochs'], callbacks = [earlystopper])

	tresults = best_cnn.evaluate(test_x, test_y,
	batch_size = params['batch_size'],
	verbose = 1,
	sample_weight = None)


	print("\t\t\t[loss, acc]")
	print(tresults)

	return [params, tresults]
  



def train_gated_cnn(train_x, train_y,test_x, test_y, params, params_tuning):

	if params_tuning == 'yes':

		gated_cnn_model = KerasClassifier(build_fn=create_gated_cnn, verbose=0)


		grid = GridSearchCV(estimator=gated_cnn_model, param_grid=params, n_jobs=-1, cv=5)

		grid_model = grid.fit(train_x, train_y)

		params = grid_model.best_params_

	best_gated_cnn = create_gated_cnn(sample_dim=params['sample_dim'], kernel_size=params['kernel_size'], layers=params['layers'], lr=params['lr'])


	earlystopper = EarlyStopping(monitor = 'accuracy', 
								patience = 40,
								min_delta = 0,
								verbose = 1,
								mode = 'auto')


	best_gated_cnn.fit(train_x, train_y, epochs=params['epochs'], batch_size=params['batch_size'], callbacks = [earlystopper])	

	tresults = best_gated_cnn.evaluate(test_x, test_y,
	batch_size = params['batch_size'],
	verbose = 1,
	sample_weight = None)


	print("\t\t\t[loss, acc]")
	print(tresults)				

	return [params, tresults]


def train_model(model_type, train_x, train_y, val_x, val_y, test_x, test_y, params, params_tuning):

	mcp = ModelCheckpoint(filepath = 'results' + "/CNNonRaw_" + str(os.getpid()) + ".hdf5",
				verbose = 0,
				save_best_only = True)


	earlystopper = EarlyStopping(monitor = 'val_loss', 
					patience = 10,
					min_delta = 0,
					verbose = 1,
					mode = 'auto')

	csv_logger = CSVLogger('results' + "/CNNonRaw_" + str(os.getpid()) + ".log.csv", 
				append=True, 
				separator='\t')
	
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
								factor=0.2,
                              	patience=5, 
								cooldown=1,
								min_lr=0.00001)

	


	if params_tuning == 'yes':
		for tmp_epochs in params['epochs']:
			for tmp_batch_size in params['batch_size']:
				for tmp_lr in params['lr']:
					for layers, flt in zip(params['layers'], params['flt']):
						model = create_model(model_type, sample_dim=params['sample_dim'], kernel_size=params['kernel_size'], flt=flt, lr=tmp_lr, layers=layers, k=params['k'])

						print("\t\t\tTraining network.")
						# don't forget to add early stopping
						history = model.fit(train_x, train_y, validation_data = (val_x, val_y), shuffle=True, epochs=tmp_epochs, batch_size=tmp_batch_size, callbacks = [ earlystopper, reduce_lr, mcp ], verbose=1)




						print("\t\t\tTesting network.")
						tresults = model.evaluate(test_x, test_y,
						batch_size = tmp_batch_size,
						verbose = 1,
						callbacks=[csv_logger],
						sample_weight = None)
						print("\t\t\t[loss, acc]")
						print(tresults)
			
						# summarize history for accuracy  
						plt.plot(history.history['accuracy'])
						plt.plot(history.history['val_accuracy'])
						plt.ylim(0.0, 1.0)
						plt.title('Model Accuracy')
						plt.ylabel('Accuracy')
						plt.xlabel('Epoch')
						plt.legend(['Train', 'Validation'], loc='lower right')
						plt.savefig('results' + "/CNNonRaw.acc.png", dpi=300)
						plt.clf()

						# summarize history for loss
						plt.plot(history.history['loss'])
						plt.plot(history.history['val_loss'])
						plt.ylim(0.0, max(max(history.history['loss']), max(history.history['val_loss'])))
						plt.title('Model Loss')
						plt.ylabel('Categorical Crossentropy')
						plt.xlabel('Epoch')
						plt.legend(['Train', 'Validation'], loc='upper right')
						plt.savefig('results' + "/CNNonRaw.loss.png", dpi=300)
						plt.clf()



						f.write('batch_size: ', tmp_batch_size, ' epochs: ', tmp_epochs, ', tmp_lr: ', lr,  ', flt: ', flt, '\n')	
						f.write(tresults, '\n\n')


	else:  ##param tuning= off

		print ("\t\t\tTraining network.")


		# train_x = [v for k, v in train_x.items()]
		# val_x = [v for k, v in val_x.items()]
		# test_x = [v for k, v in test_x.items()]


		# print(type(train_x), type(train_x[0]), len(train_x[0]))
		# exit()

		model = create_model(model_type, sample_dim=params['sample_dim'], kernel_size=params['kernel_size'], flt=params['flt'], lr=params['lr'], layers=params['layers'], k=params['k'])
		
		model.fit(train_x, train_y, validation_data = (val_x, val_y), shuffle=True, epochs=params['epochs'], batch_size=params['batch_size'], callbacks = [earlystopper, csv_logger, mcp, reduce_lr], verbose=1)

		model.save('results/saved_model.h5')

		print("\n\t\t\t\tEvaluation: [loss, acc]\n")

		# print(test_x[3].shape, test_y[3].shape, params['sample_dim'][3])
		# print(test_x[4].shape, test_y[4].shape, params['sample_dim'][4])
		# print(test_x[5].shape, test_y[5].shape, params['sample_dim'][5])


		tresults = model.evaluate(test_x, test_y,
										batch_size = params['batch_size'],
										verbose = 1,
										sample_weight = None)
		
		print(params)
		print(tresults)

		return [model, params, tresults]

def train_model_one_hot(model_type, train_x, train_y, val_x, val_y, test_x, test_y, params, params_tuning):

	mcp = ModelCheckpoint(filepath = 'results' + "/CNNonRaw_" + str(os.getpid()) + ".hdf5",
				verbose = 0,
				save_best_only = True)


	earlystopper = EarlyStopping(monitor = 'val_loss', 
					patience = 10,
					min_delta = 0,
					verbose = 1,
					mode = 'auto')

	csv_logger = CSVLogger('results' + "/CNNonRaw_" + str(os.getpid()) + ".log.csv", 
				append=True, 
				separator='\t')
	
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
								factor=0.2,
                              	patience=5, 
								cooldown=1,
								min_lr=0.00001)

	


	if params_tuning == 'yes':
		for tmp_epochs in params['epochs']:
			for tmp_batch_size in params['batch_size']:
				for tmp_lr in params['lr']:
					for layers, flt in zip(params['layers'], params['flt']):
						model = create_model(model_type, sample_dim=params['sample_dim'], kernel_size=params['kernel_size'], flt=flt, lr=tmp_lr, layers=layers, k=params['k'])

						print("\t\t\tTraining network.")
						# don't forget to add early stopping
						history = model.fit(train_x, train_y, validation_data = (val_x, val_y), shuffle=True, epochs=tmp_epochs, batch_size=tmp_batch_size, callbacks = [ earlystopper, reduce_lr, mcp ], verbose=1)




						print("\t\t\tTesting network.")
						tresults = model.evaluate(test_x, test_y,
						batch_size = tmp_batch_size,
						verbose = 1,
						callbacks=[csv_logger],
						sample_weight = None)
						print("\t\t\t[loss, acc]")
						print(tresults)
			
						# summarize history for accuracy  
						plt.plot(history.history['accuracy'])
						plt.plot(history.history['val_accuracy'])
						plt.ylim(0.0, 1.0)
						plt.title('Model Accuracy')
						plt.ylabel('Accuracy')
						plt.xlabel('Epoch')
						plt.legend(['Train', 'Validation'], loc='lower right')
						plt.savefig('results' + "/CNNonRaw.acc.png", dpi=300)
						plt.clf()

						# summarize history for loss
						plt.plot(history.history['loss'])
						plt.plot(history.history['val_loss'])
						plt.ylim(0.0, max(max(history.history['loss']), max(history.history['val_loss'])))
						plt.title('Model Loss')
						plt.ylabel('Categorical Crossentropy')
						plt.xlabel('Epoch')
						plt.legend(['Train', 'Validation'], loc='upper right')
						plt.savefig('results' + "/CNNonRaw.loss.png", dpi=300)
						plt.clf()



						f.write('batch_size: ', tmp_batch_size, ' epochs: ', tmp_epochs, ', tmp_lr: ', lr,  ', flt: ', flt, '\n')	
						f.write(tresults, '\n\n')


	else:  ##param tuning= off

		print ("\t\t\tTraining network.")


		# train_x = [v for k, v in train_x.items()]
		# val_x = [v for k, v in val_x.items()]
		# test_x = [v for k, v in test_x.items()]


		print(type(train_x), type(train_x[0]), len(train_x[0]))
		train_x=list(train_x)
		train_x=[train_x]
		# print(len(train_x))
		# print(len(train_x[0]))
		# print(len(train_x[0][0]))
		# print(len(train_x[0][0][0]))
		# exit()
		# print ("call create_model_one_hot")
		model = create_model_one_hot(model_type, sample_dim=params['sample_dim'], kernel_size=params['kernel_size'], flt=params['flt'], lr=params['lr'], layers=params['layers'])
		
		model.fit(train_x, train_y, validation_data = (val_x, val_y), shuffle=True, epochs=params['epochs'], batch_size=params['batch_size'], callbacks = [earlystopper, csv_logger, mcp, reduce_lr], verbose=1)

		model.save('results/saved_model.h5')

		print("\n\t\t\t\tEvaluation: [loss, acc]\n")

		# print(test_x[3].shape, test_y[3].shape, params['sample_dim'][3])
		# print(test_x[4].shape, test_y[4].shape, params['sample_dim'][4])
		# print(test_x[5].shape, test_y[5].shape, params['sample_dim'][5])


		tresults = model.evaluate(test_x, test_y,
										batch_size = params['batch_size'],
										verbose = 1,
										sample_weight = None)
		
		print(params)
		print(tresults)

		return [model, params, tresults]

