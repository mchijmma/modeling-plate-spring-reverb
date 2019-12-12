from __future__ import division
from __future__ import print_function

import sys
import os

from sacred import Experiment
from Config import config_ingredient
import Models
from Layers import Generator, GeneratorContextSpec, preProcessingData
import Utils
import librosa
import numpy as np
import math
import json

from __main__ import kSR, kN_dft, kN_hop, kContext

def evaluate(cfg, model_type, nameModel):
   
    model_config = cfg[model_type]

    print('Evaluating ', model_config['modelName'], 'type: ', model_type)

    # Xtest, Ytestshould be tensors of shape (number_of_recordings, number_of_samples, 1) 
    Xtest = np.random.rand(1, 2*kSR, 1)
    Ytest = np.random.rand(1, 2*kSR, 1)
    
    # zero pad at the end as well. 
    Xtest = Utils.cropAndPad(Xtest, crop = 0, pad = kContext*model_config['winLength']//2)
    Ytest = Utils.cropAndPad(Ytest, crop = 0, pad = kContext*model_config['winLength']//2)
    
    YDtest, _ = preProcessingData(Ytest,  model_config['winLength'], model_config['winLength']//2,
                      window = True, preEmphasis = False,
                      spec = True, n_fft = kN_dft, n_hop = kN_hop, log = True, power = 2.0)
    
    Ytest_w = preProcessingData(Ytest,  model_config['winLength'], model_config['winLength']//2,
                          window = True, preEmphasis = False,
                          spec = False)

    
    kLen = Xtest.shape[1]
    kBatch = int((kLen/(model_config['winLength']//2)) + 1)

    if model_type in 'model_1':

        model = Models.model_1(model_config['winLength'],
                            model_config['filters'], 
                            model_config['kernelSize'], 
                            model_config['learningRate'], kBatch)

        testGen = GeneratorContextSpec(Xtest, [YDtest, Ytest_w], kContext,
                                    model_config['winLength'], model_config['winLength']//2,
                                    kN_dft, kN_hop, win = False, win_input=False)

    elif model_type in 'model_2':

        model = Models.model_2(model_config['winLength'],
                            model_config['filters'], 
                            model_config['kernelSize'], 
                            model_config['learningRate'])

        testGen = GeneratorContextSpec(Xtest, [YDtest, Ytest_w], kContext,
                                    model_config['winLength'], model_config['winLength']//2,
                                    kN_dft, kN_hop, win = False, win_input=False)



    # load trained model
    
    model.load_weights(model_config['modelsPath']+nameModel+'_chk.h5', by_name=True) 
    
    if os.path.isdir('./Audio_'+nameModel) == False:
        os.mkdir('./Audio_'+nameModel)
   
    metrics = {}
    scores = []
    for idx in range(Xtest.shape[0]):

        x = testGen[idx][0]
        y = testGen[idx][1]
        
        
        scores_ = model.evaluate(x, y, batch_size=kBatch)
        scores.append(scores_)
        
        Z = model.predict(x, batch_size=kBatch, verbose=1)[1]
        
        Ztest_waveform = Utils.overlap(Z, kLen,
                                       model_config['winLength'], model_config['winLength']//2, windowing=False, rate=2)

        librosa.output.write_wav('./Audio_'+nameModel+'/'+nameModel+'_'+str(idx)+'.wav',
                             Ztest_waveform, kSR, norm=False)
                
    losses = {}
    for l in range(Xtest.shape[0]):
        for i,j in enumerate(model.metrics_names):
            losses.setdefault(j, []).append(scores[l][i])  

    for i,j in enumerate(model.metrics_names):
        metrics[j] = round(np.mean(losses[j]),5)

    for metric in metrics.items():
        print(metric)
        
    with open(model_config['modelsPath']+nameModel+'_metrics.json', 'w') as outfile:
        json.dump(metrics, outfile)
        
 
    print('Evaluation finished.')
