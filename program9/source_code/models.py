import warnings
warnings.filterwarnings("ignore", message=".*Your Warning Message Here.*")

import numpy as np
import random as rn
import sys, re
import keras as kr
from math import sqrt
import os


from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization, ZeroPadding1D, ZeroPadding2D,Conv2D,MaxPooling2D,concatenate
from keras.layers import LeakyReLU
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input
from keras.activations import sigmoid, softmax
import tensorflow as tf



np.random.seed(2000)

# Necessary for starting core Python generated random numbers in a well-defined state.
rn.seed(2023)

def DeepRfam(seq_length,num_c, num_filters=256,
             filter_sizes=[24, 36, 48, 60, 72, 84, 96, 108],
             dropout_rate=0.5, num_classes=1, num_hidden=512):
    # initialization

    input_shape = Input(shape=(seq_length, num_c, 1))  #  input_shape = Input(shape = (100, 4, 1))

    pooled_outputs = []
    for i in range(len(filter_sizes)):
        # print (seq_length,num_c, filter_sizes[i])
        conv = Conv2D(num_filters, (filter_sizes[i], num_c), padding='valid', activation='relu')(input_shape)
        # print ("(*)")
        pool = MaxPooling2D((seq_length - filter_sizes[i] + 1, 1), padding='valid')(conv)
        pooled_outputs.append(pool)

    merge = concatenate(pooled_outputs)

    x = Flatten()(merge)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_hidden, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    out = Dense(num_classes, activation='sigmoid')(x)

    model = Model(input_shape, out)
    
  #   out = Dense(units = sqrt(x.shape[1]))(x)
  #   out = LeakyReLU()(out)
  #   out = BatchNormalization()(out)
  #   out = Dropout(rate = 0.2, noise_shape = None, seed = None)(out)
  #   # # #out = Dense(units = 50, kernel_initializer = "he_normal", kernel_regularizer = regularizers.l2(0.00001))(out)
  #   out = Dense(units = sqrt(out.shape[1]))(out)
  #   out = LeakyReLU()(out)
  #   out = BatchNormalization()(out)
  #   out = Dropout(rate = 0.2, noise_shape = None, seed = None)(out)

  # out = Dense(units = 1, activation = "sigmoid")(out)  
    return model

def cnn1(input_sequence,kernel_Size=5,flt=70):
  print('size of input sequence', np.shape(input_sequence))
  print ("kernel_Size=",kernel_Size)
  print ("filters=",flt)

  x = Conv1D(filters = 32, kernel_size = 5, strides = 1, padding = "same")(input_sequence)
  x = LeakyReLU()(x)
  x = Conv1D(filters = 64, kernel_size = 30, strides = 1, padding = "same")(input_sequence)
  x = LeakyReLU()(x)
    # x = Dropout(rate = 0.2, noise_shape = None, seed = None)(x)
  out = Flatten()(x) 
  return out

def cnn(input_sequence,kernel_Size=5,flt=70):
  mode=2
  print ("mode=",mode)
  print('size of input sequence', np.shape(input_sequence))
  print ("kernel_Size=",kernel_Size)
  print ("filters=",flt)

  if (mode==0):
    x = Conv1D(filters = 32, kernel_size = 5, strides = 1, padding = "same")(input_sequence)
    x = LeakyReLU()(x)

    x = Conv1D(filters = 64, kernel_size = 30, strides = 1, padding = "same")(input_sequence)
    x = LeakyReLU()(x)


    # x = Dropout(rate = 0.2, noise_shape = None, seed = None)(x)
    out = Flatten()(x) 
  
  if (mode==1):
    x = Conv1D(filters = flt, kernel_size = kernel_Size, strides = 1, padding = "same")(input_sequence)
    x = LeakyReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(filters = 2*flt, kernel_size = kernel_Size, strides = 1, padding = "same")(input_sequence)
    x = LeakyReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # x = Dropout(rate = 0.2, noise_shape = None, seed = None)(x)
    out = Flatten()(x)
    
  if (mode==2): ##vasilikis setup

    x = Conv1D(filters = flt, kernel_size = kernel_Size, strides = 1, padding = "same")(input_sequence)

    x = LeakyReLU()(x)
    # print('before batchnormalization', x.shape)
    x = BatchNormalization(renorm=True)(x)
    # x = MaxPooling2D(pool_size = 3, padding = "same")(x)
    # print('after normalization', x.shape)

    x = Dropout(rate = 0.2, noise_shape = None, seed = None)(x)

    x = Conv1D(filters = flt*1.5, kernel_size = kernel_Size, strides = 1, padding = "same")(x)  ##flt=70
    x = LeakyReLU()(x)
    x = BatchNormalization(renorm=True)(x)
    # x = MaxPooling2D(pool_size = 3, padding = "same")(x)
    x = Dropout(rate = 0.2, noise_shape = None, seed = None)(x)

    x = Conv1D(filters = flt*2, kernel_size = kernel_Size, strides = 1, padding = "same")(x) ##flt=100
    x = LeakyReLU()(x)
    x = BatchNormalization(renorm=True)(x)
    # x = MaxPooling1D(pool_size = 2, padding = "same")(x)
    x = Dropout(rate = 0.2, noise_shape = None, seed = None)(x) ##flt=150

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

  out = Dense(units = 1, activation = "sigmoid")(out)  

  return out


def create_model(model_type, sample_dim, kernel_size, flt, layers, lr, k):
  sequence_input = []
  concatenated = []
  size = len(k)
  print ("sample_dim",sample_dim)
  
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

