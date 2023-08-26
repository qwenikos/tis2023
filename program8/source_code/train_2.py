import misc
from keras.layers import Input
from models import create_cnn, create_gated_cnn, create_hybrid_model
from keras.models import Model

from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


# applies the grid-search algorithm for hyperparameter tuning in a cnn model
# , and afterward, trains the best cnn model and return it
def train_cnn(train_x, train_y, test_x, test_y, params, params_tuning):

	if params_tuning == 'yes':    

		cnn_model = KerasClassifier(build_fn=create_cnn, verbose=0)

		grid = GridSearchCV(estimator=cnn_model, param_grid=params, n_jobs=-1, cv=5)


		grid_model = grid.fit(train_x, train_y)

		params = grid_model.best_params_

		print(params)


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

		print(params)

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


def train_hybrid_model(train_x, train_y, val_x, val_y, test_x, test_y, params, params_tuning):
	
	earlystopper = EarlyStopping(monitor = 'accuracy', 
								patience = 40,
								min_delta = 0,
								verbose = 1,
								mode = 'auto')


	if params_tuning == 'yes':


					
					




		# hybrid_model = KerasClassifier(build_fn=create_hybrid_model, verbose=0)

		# print(params)

		# grid = GridSearchCV(estimator=hybrid_model, param_grid=params, n_jobs=-1, cv=(val_x, val_y))

		# grid_model = grid.fit(train_x, train_y)

		# params = grid_model.best_params_




	# earlystopper = EarlyStopping(monitor = 'accuracy', 
	# 				patience = 40,
	# 				min_delta = 0,
	# 				verbose = 1,
	# 				mode = 'auto')

	# print ("\t\t\tTraining network.")

	best_hybrid = create_hybrid_model(sample_dim=params['sample_dim'], kernel_size=params['kernel_size'], flt=params['flt'], lr=params['lr'], mntm=params['mntm'], layers=params['layers'], k=params['k'])

	# don't forget to add early stopping
	best_hybrid.fit(train_x, train_y, validation_data = (val_x, val_y), shuffle=True, epochs=params['epochs'], batch_size=params['batch_size'], callbacks = [earlystopper], verbose=1)


	print("\n\t\t\t\tEvaluation: [loss, acc]\n")

	tresults = best_hybrid.evaluate(test_x, test_y,
									batch_size = params['batch_size'],
									verbose = 1,
									sample_weight = None)


	return [params, tresults]

