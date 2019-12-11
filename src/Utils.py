from __future__ import division

import numpy as np
import pickle
import random
import librosa
import scipy
random.seed(4264523625)
from scipy.signal import lfilter



def dumpPickle(d, name, path):
    
    with open(path + name, 'wb') as output:
    # Pickle dictionary using protocol 0.
        pickle.dump(d, output)
    print('%s Saved' % (name))
    
def overlap(x, x_len, win_length, hop_length, windowing = True, rate = 1): 
    x = x.reshape(x.shape[0],x.shape[1]).T
    if windowing:
        window = scipy.signal.hann(win_length, sym=False)
        rate = rate*hop_length/win_length
    else:
        window = 1
        rate = 1
    n_frames = x_len / hop_length
    expected_signal_len = int(win_length + hop_length * (n_frames))
    y = np.zeros(expected_signal_len)
    for i in range(int(n_frames)):
            sample = i * hop_length 
            w = x[:, i]
            y[sample:(sample + win_length)] = y[sample:(sample + win_length)] + w*window
    y = y[int(win_length // 2):-int(win_length // 2)]
    return np.float32(y*rate)   


def cropAndPad(x, crop = 0, pad = None):
    X = []
    for x_ in x:
        X_ = (x_[crop:,0])
        if pad:
            zeros = np.zeros((pad,))
            X_ = np.concatenate((X_,zeros))
        X.append(X_)
    X = np.asarray(X)
    
    return X.reshape(x.shape[0],-1,1)   

# objective Metrics


# mae

def getMAEnormalized(ytrue, ypred):
    
    ratio = np.mean(np.abs(ytrue))/np.mean(np.abs(ypred))

    return mean_absolute_error(ytrue, ratio*ypred)

# mfcc_cosine


def getMFCC(x, sr, mels=40, mfcc=13, mean_norm=False):
    
    melspec = librosa.feature.melspectrogram(y=x, sr=sr, S=None,
                                     n_fft=4096, hop_length=2048,
                                     n_mels=mels, power=2.0)
    melspec_dB = librosa.power_to_db(melspec, ref=np.max)
    mfcc = librosa.feature.mfcc(S=melspec_dB, sr=sr, n_mfcc=mfcc)
    if mean_norm:
        mfcc -= (np.mean(mfcc, axis=0))
    return mfcc

        
def getMSE_MFCC(y_true, y_pred, sr, mels=40, mfcc=13, mean_norm=False):
    
    ratio = np.mean(np.abs(y_true))/np.mean(np.abs(y_pred))
    y_pred =  ratio*y_pred
    
    y_mfcc = getMFCC(y_true, sr, mels=mels, mfcc=mfcc, mean_norm=mean_norm)
    z_mfcc = getMFCC(y_pred, sr, mels=mels, mfcc=mfcc, mean_norm=mean_norm)
    
    return getDistances(y_mfcc[:,:], z_mfcc[:,:]) 

