# -*- coding: utf-8 -*-
"""
Script to learn a model with Scikit-learn.

Created on Mon Oct 24 20:51:47 2022

@author: ValBaron10
"""

import numpy as np
from sklearn import preprocessing
from sklearn import svm
import pickle
from joblib import dump
from features_functions import compute_features
import wave


# LOOP OVER THE SIGNALS
for i in range(10):
#for all signals:
    # Open the WAV file
    FILENAME = "signal_" + str(i*100) + ".wav"

    with wave.open("Data/{}".format(FILENAME), 'rb') as wav_file:
        # Get the parameters of the WAV file
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        num_frames = wav_file.getnframes()
        input_sig = np.frombuffer(wav_file.readframes(num_frames), dtype=np.int16)



    # Compute the signal in three domains
    sig_sq = input_sig**2
    sig_t = input_sig / np.sqrt(sig_sq.sum())
    sig_f = np.absolute(np.fft.fft(sig_t))
    sig_c = np.absolute(np.fft.fft(sig_f))

    # Compute the features and store them
    features_list = []
    N_feat, features_list = compute_features(sig_t, sig_f[:sig_t.shape[0]//2], sig_c[:sig_t.shape[0]//2])
    features_vector = np.array(features_list)[np.newaxis,:]

    # Store the obtained features in a np.arrays
    #learningFeatures = # 2D np.array with features_vector in it, for each signal
    learningFeatures = []
    learningFeatures.append(features_vector)
    # Store the labels
    #learningLabels = # np.array with labels in it, for each signal
    for a in learningFeatures :
        learningLabels = np.column_stack((learningFeatures))
        
        if FILENAME.startswith("signal_"):
            a == 0
        if FILENAME.startswith("noise") : 
            a == 1
            
        learningLabels.append(a)

# Encode the class names
labelEncoder = preprocessing.LabelEncoder().fit(learningLabels)
learningLabelsStd = labelEncoder.transform(learningLabels)

# Learn the model
model = svm.SVC(C=10, kernel='linear', class_weight=None, probability=False)
scaler = preprocessing.StandardScaler(with_mean=True).fit(learningFeatures)
learningFeatures_scaled = scaler.transform(learningFeatures)
model.fit(learningFeatures_scaled, learningLabelsStd)

# Export the scaler and model on disk
dump(scaler, "SCALER")
dump(model, "SVM_MODEL")

