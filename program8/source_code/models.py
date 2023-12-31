import numpy as np
import random as rn
import sys, re
import keras as kr
from math import sqrt

from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization, ZeroPadding1D,ZeroPadding2D,Conv2D, MaxPooling2D
from keras.layers import LeakyReLU
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input
from keras.activations import sigmoid, softmax
import tensorflow as tf


np.random.seed(2000)

# Necessary for starting core Python generated random numbers in a well-defined state.
rn.seed(2023)

def cnn(input_sequence, flt, kern_size):
  print('size of input sequence', input_sequence)
  x = Conv1D(filters = 70, kernel_size = 4, strides = 1, padding = "same")(input_sequence)
  x = LeakyReLU()(x)
  # print('before batchnormalization', x.shape)
  x = BatchNormalization(renorm=True)(x)
  # x = MaxPooling2D(pool_size = 3, padding = "same")(x)
  # print('after normalization', x.shape)

  x = Dropout(rate = 0.2, noise_shape = None, seed = None)(x)

  x = Conv1D(filters = 100, kernel_size = 4, strides = 1, padding = "same")(x)
  x = LeakyReLU()(x)
  x = BatchNormalization(renorm=True)(x)
  # x = MaxPooling2D(pool_size = 3, padding = "same")(x)
  x = Dropout(rate = 0.2, noise_shape = None, seed = None)(x)

  x = Conv1D(filters = 150, kernel_size = 4, strides = 1, padding = "same")(x)
  x = LeakyReLU()(x)
  x = BatchNormalization(renorm=True)(x)
  # x = MaxPooling1D(pool_size = 2, padding = "same")(x)
  x = Dropout(rate = 0.2, noise_shape = None, seed = None)(x)

  # x = Conv2D(filters = 180, kernel_size = 7, strides = 1, padding = "same")(x)
  # x = LeakyReLU()(x)
  # x = BatchNormalization()(x)
  # x = MaxPooling1D(pool_size = 2, padding = "same")(x)
  # x = Dropout(rate = 0.2, noise_shape = None, seed = None)(x)

  # x = Conv1D(filters = flt, kernel_size = 4, strides = 1, padding = "same")(x)
  # x = LeakyReLU()(x)
  # x = BatchNormalization()(x)
  # x = MaxPooling1D(pool_size = 2, padding = "same")(x)
  # x = Dropout(rate = 0.2, noise_shape = None, seed = None)(x)    

  out = Flatten()(x)

  return out
  

  
# initializes and returns a cnn model
def create_cnn(sample_dim, lr, mntm, flt):
  sequence_input = Input(shape = (sample_dim[0], sample_dim[1], sample_dim[2]))

  out = cnn(sequence_input, flt)

  cnn_model = Model(sequence_input, out)

  print(lr)
  sgd = SGD(learning_rate = lr,
    decay = 1e-6,
    momentum = mntm,
    nesterov = True)

  cnn_model.compile(optimizer = sgd,
    loss = "binary_crossentropy",
    metrics = ["accuracy"])
  
  return cnn_model


def gated_cnn(sequence, kern_size, layers):

  input_sequence = ZeroPadding2D((kern_size-1,0))(sequence)

  filters = 70

  for i in range(3):

    c1 = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = 'same')(input_sequence)
    b = np.random.standard_normal((1, 1, c1.shape[2]))
    b = np.tile(b, (1, c1.shape[1], 1))

    A = c1 + b

    c2 = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = 'same')(input_sequence)
    d= np.random.standard_normal((1, 1 , c2.shape[2]))
    d= np.tile(d, (1, c2.shape[1], 1))

    B = c2 + d

    h = A * sigmoid(B)

    input_sequence = h

    filters += 30

  out = Flatten()(input_sequence)

  return out
  

# initializes and returns a cnn model
def create_gated_cnn(sample_dim, kernel_size, layers, lr, mntm=0.9):
  sequence_input = Input(shape = (sample_dim[0], sample_dim[1]))

  out = gated_cnn(sequence_input, kernel_size, layers)

  model = Model(sequence_input, out)

  sgd = SGD(learning_rate = lr,
    decay = 1e-6,
    momentum = mntm,
    nesterov = True)

  model.compile(loss = "binary_crossentropy",
                optimizer='sgd',
                metrics = ["accuracy"])
  
  return model

def feature_extraction(model_type, input_sequence, kern_size, flt, layers):

  if model_type == 'hybrid':
    x_cnn = cnn(input_sequence, flt, kern_size)
    x_gated_cnn = gated_cnn(input_sequence, kern_size, layers)

    out = tf.concat((x_cnn, x_gated_cnn), axis=1)

    return out
  
  elif model_type == 'cnn':
    out = cnn(input_sequence, flt, kern_size)

    return out

  elif model_type == 'gated_cnn':
    out = gated_cnn(input_sequence, kern_size, layers)

    return out


def classification(x):
  #out = Dense(units = 100, kernel_initializer = "he_normal", kernel_regularizer = regularizers.l2(0.00001))(out)
  out = Dense(units = sqrt(x.shape[1]))(x)
  out = LeakyReLU()(out)
  out = BatchNormalization()(out)
  out = Dropout(rate = 0.2, noise_shape = None, seed = None)(out)
  # # #out = Dense(units = 50, kernel_initializer = "he_normal", kernel_regularizer = regularizers.l2(0.00001))(out)
  out = Dense(units = sqrt(out.shape[1]))(out)
  out = LeakyReLU()(out)
  out = BatchNormalization()(out)
  out = Dropout(rate = 0.2, noise_shape = None, seed = None)(out)
  # out = Dense(units = 150)(out)
  # out = LeakyReLU()(out)
  # out = BatchNormalization()(out)
  # out = Dropout(rate = 0.25, noise_shape = None, seed = None)(out)
  # out = Dense(units = 50)(out)
  # out = LeakyReLU()(out)
  # out = BatchNormalization()(out)
  # out = Dropout(rate = 0.2, noise_shape = None, seed = None)(out)
  # out = Dense(units = 50)(out)
  # out = LeakyReLU()(out)
  # out = BatchNormalization()(out)
  # out = Dropout(rate = 0.2, noise_shape = None, seed = None)(out)
  # out = Dense(units = 50)(out)
  # out = LeakyReLU()(out)
  # out = BatchNormalization()(out)
  # out = Dropout(rate = 0.2, noise_shape = None, seed = None)(out)
  # out = Dense(units = 100)(out)
  # out = LeakyReLU()(out)
  # out = BatchNormalization()(out)
  # out = Dropout(rate = 0.2, noise_shape = None, seed = None)(out)
  # out = Dense(units = 75)(x)
  # out = LeakyReLU()(out)
  # out = BatchNormalization()(out)
  # out = Dropout(rate = 0.2, noise_shape = None, seed = None)(out)
  # out = Dense(units = 50)(out)
  # out = LeakyReLU()(out)
  # out = BatchNormalization()(out)
  # out = Dropout(rate = 0.2, noise_shape = None, seed = None)(out)

  out = Dense(units = 1, activation = "sigmoid")(out)  

  return out


def create_model(model_type, sample_dim, kernel_size, flt, layers, lr, k):
  sequence_input = []
  concatenated = []
  size = len(k)

  for i,j in zip(k,range(size)):
    print('input_shape', sample_dim[i][0], sample_dim[i][1])
    sequence_input.append(Input(shape = (sample_dim[i][0], sample_dim[i][1])))      # upstream
    res = feature_extraction(model_type, sequence_input[j], kernel_size, flt, layers)

    print('output_shape', res.shape)

    concatenated.append(res)

  
  out = tf.concat([i for i in concatenated], axis=1)

  print(out.shape)

  # exit()
  out = classification(out)

  if size == 1:
    model = Model(inputs=sequence_input[0], outputs=out)

  else:
    model = Model(inputs=sequence_input, outputs=out)


  sgd = SGD(learning_rate = lr,
    decay = 1e-6,
    momentum = 0.9,
    nesterov = True)

  model.compile(loss = "binary_crossentropy",
                optimizer=sgd,
                metrics = ["accuracy"])

  
  return model

  

