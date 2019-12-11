from __future__ import division
from __future__ import print_function

import sys
import os

from sacred import Experiment
from Config import config_ingredient
import Models
from Layers import Generator, GeneratorContextSpec, preProcessingData
import Utils

ex = Experiment('model training', ingredients=[config_ingredient])

import numpy as np
import math
 

kSR = 16000
# constants for spectograms:
kN_dft = 4096
kN_hop = 4096
# constants context frames:
kContext = 4

import Evaluate

# model_type should be 'pretraining', 'model_1' or 'model_2'


@ex.automain
def main(cfg, model_type):
   
    model_config = cfg[model_type]
    
    if os.path.isdir(model_config['modelsPath']) == False:
        os.mkdir(model_config['modelsPath'])

    i = 0
    while os.path.exists(model_config['modelsPath']+'model_%s.pkl' % i):
        i += 1
    model_config['modelName'] = 'model_%s' % i

    Utils.dumpPickle(model_config, model_config['modelName']+ '.pkl', model_config['modelsPath'])
    
    try:

        print('Training ', model_config['modelName'])

        # Xtrain, Ytrain, Xval, Yval should be tensors of shape (number_of_recordings, number_of_samples, 1) 
        Xtrain = np.random.rand(1, 32000, 1)
        Ytrain = np.random.rand(1, 32000, 1)
        Xval = np.random.rand(1, 32000, 1)
        Yval = np.random.rand(1, 32000, 1)
        
        # since the samples are 2 secs long, we zero pad 4*hop_size samples at the end of the recording. This for the 4 
        # subsequent frames in the Leslie modeling tasks.
        Xtrain = Utils.cropAndPad(Xtrain, crop = 0, pad = kContext*model_config['winLength']//2)
        Ytrain = Utils.cropAndPad(Ytrain, crop = 0, pad = kContext*model_config['winLength']//2)
        Xval = Utils.cropAndPad(Xval, crop = 0, pad = kContext*model_config['winLength']//2)
        Yval = Utils.cropAndPad(Yval, crop = 0, pad = kContext*model_config['winLength']//2)
        
        kLen = Xtrain.shape[1]
        kBatch = int((kLen/(model_config['winLength']//2)) + 1)
            
        YDtrain, _ = preProcessingData(Ytrain,  model_config['winLength'], model_config['winLength']//2,
                      window = True, preEmphasis = False,
                      spec = True, n_fft = kN_dft, n_hop = kN_hop, log = True, power = 2.0)
    
        Ytrain_w = preProcessingData(Ytrain,  model_config['winLength'], model_config['winLength']//2,
                          window = True, preEmphasis = False,
                          spec = False)



        YDval, _ = preProcessingData(Yval, model_config['winLength'], model_config['winLength']//2,
                          window = True, preEmphasis = False,
                          spec = True, n_fft = kN_dft, n_hop = kN_hop, log = True, power = 2.0)

        Yval_w = preProcessingData(Yval, model_config['winLength'], model_config['winLength']//2,
                          window = True, preEmphasis = False,
                          spec = False)


        if model_type in 'pretraining':

            model = Models.pretrainingModel(model_config['winLength'],
                                model_config['filters'], 
                                model_config['kernelSize'], 
                                model_config['learningRate'])

            Xtrain = np.vstack((Xtrain, Ytrain))
            Xval = np.vstack((Xval, Yval))
            trainGen = Generator(Xtrain, Xtrain, model_config['winLength'], model_config['winLength']//2)
            valGen = Generator(Xval, Xval, model_config['winLength'], model_config['winLength']//2)

        elif model_type in 'model_1':

            model = Models.model_1(model_config['winLength'],
                                model_config['filters'], 
                                model_config['kernelSize'], 
                                model_config['learningRate'], kBatch)

            trainGen = GeneratorContextSpec(Xtrain, [YDtrain, Ytrain_w], kContext,
                                        model_config['winLength'], model_config['winLength']//2,
                                        kN_dft, kN_hop, win = False, win_input=False)
            
            valGen = GeneratorContextSpec(Xval, [YDval, Yval_w], kContext,
                                        model_config['winLength'], model_config['winLength']//2,
                                        kN_dft, kN_hop, win = False, win_input=False)

        elif model_type in 'model_2':

            model = Models.model_2(model_config['winLength'],
                                model_config['filters'], 
                                model_config['kernelSize'], 
                                model_config['learningRate'])

            trainGen = GeneratorContextSpec(Xtrain, [YDtrain, Ytrain_w], kContext,
                                        model_config['winLength'], model_config['winLength']//2,
                                        kN_dft, kN_hop, win = False, win_input=False)
            
            valGen = GeneratorContextSpec(Xval, [YDval, Yval_w], kContext,
                                        model_config['winLength'], model_config['winLength']//2,
                                        kN_dft, kN_hop, win = False, win_input=False)

        

        model.summary()

        # load pretrained model if available:
    #         model.load_weights(path_preatrained_model, by_name=True) 



        earlyStopping = Models.EarlyStopping(monitor=model_config['monitorLoss'],
                                          min_delta=0,
                                          patience=25,
                                          verbose=1,
                                          mode='auto',
                                          baseline=None, restore_best_weights=False)

        checkpointer = Models.ModelCheckpoint(filepath=model_config['modelsPath']+model_config['modelName']+'_chk.h5',
                                           monitor=model_config['monitorLoss'],
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=True)

        model.fit_generator(trainGen,
                           steps_per_epoch=None,
                           epochs=model_config['epoch'],
                           verbose=2,
                           callbacks = [checkpointer, earlyStopping],
                           validation_data = valGen,
                           validation_steps=len(Xval),
                           shuffle=True)

        print('Reducing Learning rate by 4')

        symbolic_weights = getattr(model.optimizer, 'weights')
        weight_values = Models.K.batch_get_value(symbolic_weights)

        model.compile(loss='mae',
                      optimizer=Models.Adam(lr=model_config['learningRate']/4))

        model.load_weights(model_config['modelsPath']+model_config['modelName']+'_chk.h5', by_name=True)

        model.fit_generator(trainGen,
                           steps_per_epoch=None,
                           epochs=model_config['epoch'],
                           verbose=2,
                           callbacks = [checkpointer, earlyStopping],
                           validation_data = valGen,
                           validation_steps=len(Xval),
                           shuffle=True)

        print('Training finished.')
        
        if model_type in ['model_1','model_2']:
            Evaluate.evaluate(cfg, model_type, model_config['modelName'])

    except Exception as e: 
        print(e)
        print('Training failed: ' + model_config['modelName'] + ' was removed')  
        os.remove(model_config['modelsPath']+model_config['modelName']+'.pkl')
    

    
    


   