from __future__ import division

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
import keras
from keras import backend as K
K.set_session(sess)

from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Activation, Concatenate, TimeDistributed, Lambda, Reshape
from keras.layers import Multiply, Add, UpSampling1D, MaxPooling1D, Bidirectional, LSTM, GlobalAvgPool1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from Layers import Conv1D_local, SAAF, Conv1D_tied, VelvetNoise, Conv1D_localTensor
from kapre.time_frequency import Spectrogram

import numpy as np
import scipy


# constants

kSR = 16000
kContext = 4
kBatch = 20
kWin = 4096
# custom general functions:
def Window(x):
    
    window = scipy.signal.hann(kWin, sym=False)
    window = K.cast(window, dtype='float32')
    window = K.reshape(window, (-1, kWin, 1))
    output = K.tf.multiply(x, window)
    
    return output

def BooleanMask(x):
  
    output = K.greater_equal(x[0], x[1])
    output = K.cast(output, dtype='float32')
    output = K.tf.multiply(output, x[1])

    return output

def toPermuteDimensions(x):
    
    return K.permute_dimensions(x, (0, 2, 1))

def MAE(y_true, y_pred):
    
    y_pred = K.squeeze(y_pred, axis = -1)
    y_true = K.squeeze(y_true, axis = -1)
    mae = K.mean(K.abs(y_pred - y_true), axis=-1)
    mae = K.mean(mae, axis = -1)
    mae = K.mean(mae, keepdims=True)
                
    return mae

def PreEmphasis(x):
    
    x_ = keras.layers.Cropping1D(cropping=(0, 1))(x)
    paddings = tf.constant([[0, 0,], [1, 0], [0, 0]])
    x_ = tf.pad(x_, paddings, "CONSTANT")  
    x_ = tf.scalar_mul(0.95, x_)   
    
    return tf.subtract(x, x_)

def MAE_preEmphasis(y_true, y_pred):
    
    y_true = PreEmphasis(y_true)
    y_pred = PreEmphasis(y_pred)
    
    return MAE(y_true, y_pred)

# model to train when initializing Conv1d and Conv1D-Local layers.

def pretrainingModel(win_length, filters, kernel_size_1, learning_rate):

    x = Input(shape=(win_length, 1), name='input')
    conv = Conv1D(filters, kernel_size_1, strides=1, padding='same',
                       kernel_initializer='lecun_uniform',
                       input_shape=(win_length, 1), name='conv')

    conv_smoothing = Conv1D_local(filters, kernel_size_1*2, strides=1, padding='same',
                                  kernel_initializer='lecun_uniform', name='conv_smoothing')

    deconv = Conv1D_tied(1, kernel_size_1, conv, padding='same', name='deconv')
    
     
    X = conv(x)
    X_abs = Activation(K.abs, name='conv_activation')(X)
    M = conv_smoothing(X_abs)
    M = Activation('softplus', name='conv_smoothing_activation')(M)
    P = X
    Z = MaxPooling1D(pool_size=win_length//64, name='max_pooling')(M)
    M_ = UpSampling1D(size=win_length//64, name='up_sampling_naive')(Z)   
    M_ = Lambda((BooleanMask), name='boolean_mask')([M,M_])
        
    Y = Multiply(name='phase_unpool_multiplication')([P,M_])
    Y = deconv(Y)
    
    model = Model(inputs=[x], outputs=[Y])
    
    model.compile(loss={'deconv': 'mae'},
                         loss_weights={'deconv': 1.0},
                        optimizer=Adam(lr=learning_rate))

    return model


# SE blocks:


def se_block(x, num_features, weight_decay=0., amplifying_ratio=16, idx = 1):
    x = Multiply(name='dnn-saaf-se_%s'%idx)([x, se_fn(x, amplifying_ratio, idx)])
    return x

def se_fn(x, amplifying_ratio, idx):
    num_features = x.shape[-1].value
    x = Activation(K.abs)(x)
    x = GlobalAvgPool1D()(x)
    x = Reshape((1, num_features))(x)
    x = Dense(num_features * amplifying_ratio, activation='relu', kernel_initializer='glorot_uniform',
              name='se_dense1_%s'%idx)(x)
    x = Dense(num_features, activation='sigmoid', kernel_initializer='glorot_uniform',
              name='se_dense2_%s'%idx)(x)
    return x

def se_block_lstm(x, num_features, weight_decay=0., amplifying_ratio=16, idx = 1):
    x = Multiply(name='dnn-saaf-se_%s'%idx)([x, se_fn_lstm(x, amplifying_ratio, idx)])
    return x

def se_fn_lstm(x, amplifying_ratio, idx):
    num_features = x.shape[-1].value
    x = Activation(K.abs)(x)
    x = GlobalAvgPool1D()(x)
    x = Reshape((1, num_features))(x) #for model AET_Convolution_16 change lstm units for num_features*ampliphying ratio
    x = LSTM(num_features, activation='relu', stateful=False,
                                 return_sequences=True, dropout=0.1,
                                 recurrent_dropout=0.1, name='se_lstm1_%s'%idx)(x)
    x = Dense(num_features * amplifying_ratio, activation='relu', kernel_initializer='glorot_uniform',
              name='se_dense1_%s'%idx)(x)
    x = Dense(num_features, activation='sigmoid', kernel_initializer='glorot_uniform',
              name='se_dense2_%s'%idx)(x)
    return x





# model-1

def model_1(win_length, filters, kernel_size_1, learning_rate, batch):
    
    
    kPs = int((win_length*2000/kSR))
    kN = int(win_length)
    
    ini1 = tf.initializers.random_uniform(minval=-1,maxval=1)
    ini2 = tf.initializers.random_uniform(minval=0,maxval=1)
    
    x = Input(shape=(kContext*2+1, win_length, 1), name='input', batch_shape=(batch, kContext*2+1, win_length, 1))
    
    conv = Conv1D(filters, kernel_size_1, strides=1, padding='same',
                       kernel_initializer='lecun_uniform', input_shape=(win_length, 1))
    
    activation_abs = Activation(K.abs)
    activation_sp = Activation('softplus')
    max_pooling = MaxPooling1D(pool_size=win_length//64)

    conv_smoothing = Conv1D_local(filters, kernel_size_1*2, strides=1, padding='same',
                                  kernel_initializer='lecun_uniform')
    
    dense_sgn = Dense(kPs, activation='tanh', kernel_initializer=ini1, name='dense_l_sgn')
    
    dense_idx = Dense(kPs, activation='sigmoid', name='dense_l_idx')

    bi_rnn = Bidirectional(LSTM(filters*2, activation='tanh', stateful=False,
                                 return_sequences=True, dropout=0.1,
                                 recurrent_dropout=0.1), merge_mode='concat', name='birnn_in')
    bi_rnn1 = Bidirectional(LSTM(filters, activation='tanh', stateful=False,
                                 return_sequences=True, dropout=0.1,
                                 recurrent_dropout=0.1), merge_mode='concat', name='birnn_1')
    bi_rnn2 = Bidirectional(LSTM(filters//2, activation='linear', stateful=False,
                                 return_sequences=True, dropout=0.1,
                                 recurrent_dropout=0.1), merge_mode='concat', name='birnn_2')
    
    bi_rnn3 = Bidirectional(LSTM(filters//2, activation='linear', stateful=False,
                                 return_sequences=True, dropout=0.1,
                                 recurrent_dropout=0.1), merge_mode='concat', name='birnn_3')
    
    convTensors = Conv1D_localTensor(filters, win_length, kBatch, strides=1, padding='same',
                                     name='convTensors')
    
    deconv = Conv1D_tied(1, kernel_size_1, conv, padding='same', name='deconv')
    
    velvet = VelvetNoise(kPs, kBatch, input_dim=filters, input_length=win_length, name='velvet')
        
    X = TimeDistributed(conv, name='conv')(x)
    X_abs = TimeDistributed(activation_abs, name='conv_activation')(X)
    M = TimeDistributed(conv_smoothing, name='conv_smoothing')(X_abs)
    M = TimeDistributed(activation_sp, name='conv_smoothing_activation')(M)
    P = X
    Z = TimeDistributed(max_pooling, name='max_pooling')(M)
    Z = Lambda(lambda inputs: tf.unstack(inputs, num=kContext*2+1, axis=1, name='unstack2'))(Z)
    Z = Concatenate(name='concatenate')(Z)

    Z = bi_rnn(Z)
    Z1 = bi_rnn1(Z)
    Z1 = bi_rnn2(Z1)
    Z1 = SAAF(break_points=25, break_range=0.2, magnitude=100, order=2, tied_feamap=True,
            kernel_initializer = 'random_normal', name = 'saaf_1')(Z1)
    
    Z2 = bi_rnn3(Z)
    Z2 = SAAF(break_points=25, break_range=0.2, magnitude=100, order=2, tied_feamap=True,
            kernel_initializer = 'random_normal', name = 'saaf_2')(Z2)

    Z1 = Lambda((toPermuteDimensions), name='perm_1')(Z1)
                        
    sgn = dense_sgn(Z1)
    idx = dense_idx(Z1)
                            
    sgn = Lambda((toPermuteDimensions), name='perm_2')(sgn)
    idx = Lambda((toPermuteDimensions), name='perm_3')(idx)

    P = Lambda(lambda inputs: tf.unstack(inputs, num=kContext*2+1, axis=1, name='unstack'))(P)
    V = Concatenate(name='concatenate2', axis=-1)([sgn,idx])
    V = velvet(V)
    
    Y = Concatenate(name='concatenate3')([P[kContext],V])
    Y = convTensors(Y)
    Y = SAAF(break_points=25, break_range=0.2, magnitude=100, order=2, tied_feamap=True,
            kernel_initializer = 'random_normal', name = 'saaf_out_conv')(Y)
    
    M_ = UpSampling1D(size=win_length//64, name='up_sampling_naive')(Z2)
    Y = Multiply(name='phase_unpool_multiplication')([Y,M_])
    
 
    Y_ = Dense(filters, activation = 'tanh', name = 'dense_in')(Y)  
    Y_ = Dense(filters//2, activation = 'tanh', name = 'dense_h1')(Y_)   
    Y_ = Dense(filters//2, activation = 'tanh', name = 'dense_h2')(Y_)
    Y_ = Dense(filters, activation = 'linear', name = 'dense_out')(Y_)
    Y_ = SAAF(break_points=25, break_range=0.2, magnitude=100, order=2, tied_feamap=True,
            kernel_initializer = 'random_normal', name = 'saaf_out')(Y_)
    
    Y = se_block_lstm(Y, filters, weight_decay=0., amplifying_ratio=16, idx = 1)
    Y_ = se_block_lstm(Y_, filters, weight_decay=0., amplifying_ratio=16, idx = 2)
    
    Y = Add(name='addition')([Y,Y_])
    Y = deconv(Y)
    
    Y = Lambda((Window), name='waveform')(Y)


        
    loss_output = Spectrogram(n_dft=win_length, n_hop=win_length, input_shape=(1, win_length), 
      return_decibel_spectrogram=True, power_spectrogram=2.0, 
      trainable_kernel=False, name='spec')

    spec = Lambda((toPermuteDimensions), name='perm_spec')(Y)
    spec = loss_output(spec)

    model = Model(inputs=[x], outputs=[spec, Y])

    model.compile(loss={'spec': 'mse', 'waveform': MAE_preEmphasis},
                        loss_weights={'spec': 0.0001, 'waveform': 1.0},
                       optimizer=Adam(lr=learning_rate))


    return model





# model-2.

'''Architecture taken from:
    A general-purpose deep learning approach to model time-varying audio effects, Martinez Ramirez M. A., Benetos E. and  Reiss J. D., in the 22nd International Conference on Digital Audio Effects (DAFx-19), Birmingham, UK, September 2019.
    
    https://mchijmma.github.io/modeling-time-varying/
     '''

def model_2(win_length, filters, kernel_size_1, learning_rate):
   
    kContext = 4 # past and subsequent frames
    
    x = Input(shape=(kContext*2+1, win_length, 1), name='input')

    
    conv = Conv1D(filters, kernel_size_1, strides=1, padding='same',
                       kernel_initializer='lecun_uniform', input_shape=(win_length, 1))
    
    activation_abs = Activation(K.abs)
    activation_sp = Activation('softplus')
    max_pooling = MaxPooling1D(pool_size=win_length//64)

    conv_smoothing = Conv1D_local(filters, kernel_size_1*2, strides=1, padding='same',
                                  kernel_initializer='lecun_uniform')

    bi_rnn = Bidirectional(LSTM(filters*2, activation='tanh', stateful=False,
                                 return_sequences=True, dropout=0.1,
                                 recurrent_dropout=0.1), merge_mode='concat', name='birnn_in')
    bi_rnn1 = Bidirectional(LSTM(filters, activation='tanh', stateful=False,
                                 return_sequences=True, dropout=0.1,
                                 recurrent_dropout=0.1), merge_mode='concat', name='birnn_1')
    bi_rnn2 = Bidirectional(LSTM(filters//2, activation='linear', stateful=False,
                                 return_sequences=True, dropout=0.1,
                                 recurrent_dropout=0.1), merge_mode='concat', name='birnn_2')
    
    deconv = Conv1D_tied(1, kernel_size_1, conv, padding='same', name='deconv')
    
    
    X = TimeDistributed(conv, name='conv')(x)
    X_abs = TimeDistributed(activation_abs, name='conv_activation')(X)
    M = TimeDistributed(conv_smoothing, name='conv_smoothing')(X_abs)
    M = TimeDistributed(activation_sp, name='conv_smoothing_activation')(M)
    P = X
    Z = TimeDistributed(max_pooling, name='max_pooling')(M)
    Z = Lambda(lambda inputs: tf.unstack(inputs, num=kContext*2+1, axis=1, name='unstack2'))(Z)
    Z = Concatenate(name='concatenate')(Z)

    Z = bi_rnn(Z)
    Z = bi_rnn1(Z)
    Z = bi_rnn2(Z)
    Z = SAAF(break_points=25, break_range=0.2, magnitude=100, order=2, tied_feamap=True,
            kernel_initializer = 'random_normal', name = 'saaf_1')(Z)

    M_ = UpSampling1D(size=win_length//64, name='up_sampling_naive')(Z)
    P = Lambda(lambda inputs: tf.unstack(inputs, num=kContext*2+1, axis=1, name='unstack'))(P)
    Y = Multiply(name='phase_unpool_multiplication')([P[kContext],M_])
  
    Y_ = Dense(filters, activation = 'tanh', name = 'dense_in')(Y)  
    Y_ = Dense(filters//2, activation = 'tanh', name = 'dense_h1')(Y_)   
    Y_ = Dense(filters//2, activation = 'tanh', name = 'dense_h2')(Y_)
    Y_ = Dense(filters, activation = 'linear', name = 'dense_out')(Y_)
    Y_ = SAAF(break_points=25, break_range=0.2, magnitude=100, order=2, tied_feamap=True,
            kernel_initializer = 'random_normal', name = 'saaf_out')(Y_)
    
    Y_ = se_block(Y_, filters, weight_decay=0., amplifying_ratio=16, idx = 1)
    Y = Add(name='addition')([Y,Y_])
    Y = deconv(Y)
    
    Y = Lambda((Window), name='waveform')(Y)


        
    loss_output = Spectrogram(n_dft=win_length, n_hop=win_length, input_shape=(1, win_length), 
      return_decibel_spectrogram=True, power_spectrogram=2.0, 
      trainable_kernel=False, name='spec')

    spec = Lambda((toPermuteDimensions), name='perm_spec')(Y)
    spec = loss_output(spec)

    model = Model(inputs=[x], outputs=[spec, Y])

    model.compile(loss={'spec': 'mse', 'waveform': MAE_preEmphasis},
                        loss_weights={'spec': 0.0001, 'waveform': 1.0},
                       optimizer=Adam(lr=learning_rate))


    return model
    



