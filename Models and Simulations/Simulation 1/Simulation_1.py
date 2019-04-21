#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 08:31:14 2019

@author: slavi
"""
# Simulation 1 --> Replicating the Primacy, Recency, Word length effects
# Botvinick & Plaut, 2006

#  Overall accuracy for lists of six nonconfusable letters in this data set
# was 58%, and this provided the stopping criterion for our simulations


from Binding_Pool_parameters import *
from BP_STM_parameters import *
import numpy as np
from numpy import zeros, newaxis
from keras.models import load_model
import matplotlib.pyplot as plt


# LSTM model

ISR_model = load_model(
        'first_sr_model_lstm.h5', custom_objects={'my_accuracy': my_accuracy}
        )

list_len = 6
def make_input_trial():
    global letters_26
    list_len = 6
    trial_input = np.zeros(shape=(list_len * 2 + 1, len(letters_26) + 1))
    letters = np.random.permutation(len(letters_26))
    for i in range(list_len):
        #encoding
        trial_input[i, letters[i]] = 1
        #recall
        #recall cue
        trial_input[i + list_len, len(letters_26)] = 1
        #recall cue
    trial_input[list_len * 2, len(letters_26)] = 1
    
    return trial_input.reshape(1, list_len * 2 + 1, len(letters_26) + 1)
 


global position           
position = np.array([0, 0, 0, 0, 0, 0])

for _ in range(1, 1000):    
    trial = make_input_trial()
    prediction = ISR_model.predict(trial)
    prediction = prediction[0][-(list_len+1):]
    prediction = prediction[:-1]
    prediction = prediction[newaxis,:,:]
    trial = trial[0][:-(list_len+1)]
    trial = trial[newaxis,:,:]
    for pos, (position_p, position_t) in enumerate(zip(prediction[0], trial[0])):
        if np.argmax(position_p) == np.argmax(position_t):
            position[pos] += 1
            
            
# Plots the model’s accuracy on six-item lists, 
# evaluated separately for each position.
def plot_prim_lstm():
    plt.plot(position)
    plt.xlabel('Item Position')
    plt.ylabel('Accuracy')
    plt.axis([1,6,0,1000])
    plt.show()

 

#  Word length effect

def make_input_trial_len(list_len):
    global letters_26
    trial_input = np.zeros(shape=(list_len * 2 + 1, len(letters_26) + 1))
    letters = np.random.permutation(len(letters_26))
    for i in range(list_len):
        #encoding
        trial_input[i, letters[i]] = 1
        #recall
        #recall cue
        trial_input[i + list_len, len(letters_26)] = 1
        #recall cue
    trial_input[list_len * 2, len(letters_26)] = 1
    
    return trial_input.reshape(1, list_len * 2 + 1, len(letters_26) + 1)



def examples_generator_len():
    max_list_len = 9
    while(True):
        for list_len in range(1, max_list_len + 1):
            yield make_input_trial_len(list_len)
    

len_check = examples_generator_len()

global proportion
global error
error = False
proportion_errors = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

for _ in range(900):
    trial = next(len_check)
    prediction = ISR_model.predict(trial)
    prediction = prediction[0][-(
            (
                    (len(trial[0])
                    -1)
                    //2)
                    +1):]
    prediction = prediction[:-1]
    prediction = prediction[newaxis,:,:]
    trial = trial[0][:-(
            (
                    (len(trial[0])
                    -1)
                    //2)
                    +1)]
    trial = trial[newaxis,:,:]
    pos_in_proportion = len(trial[0])-1
    
    for position_p, position_t in zip(prediction[0], trial[0]):
        if np.argmax(position_p) == np.argmax(position_t):
            error = False
            
        else:
            error = True
            break

    if error:
        proportion_errors[pos_in_proportion] += 1
            
    
def plot_w_len_lstm():
    plt.plot(proportion_errors)
    plt.xlabel('Number of items in a list')
    plt.ylabel('Errors')
    plt.axis([1,9,0,150])
    plt.show()

######################################################################
#  RNN model, same simulations

# 1 a) Primacy/Recency



ISR_model_rnn = load_model(
        'first_sr_model_rnn.h5', custom_objects={'my_accuracy': my_accuracy}
        )

list_len = 6
def make_input_trial():
    global letters_26
    list_len = 6
    trial_input = np.zeros(shape=(list_len * 2 + 1, len(letters_26) + 1))
    letters = np.random.permutation(len(letters_26))
    for i in range(list_len):
        #encoding
        trial_input[i, letters[i]] = 1
        #recall
        #recall cue
        trial_input[i + list_len, len(letters_26)] = 1
        #recall cue
    trial_input[list_len * 2, len(letters_26)] = 1
    
    return trial_input.reshape(1, list_len * 2 + 1, len(letters_26) + 1)
 


global position_rnn           
position_rnn = np.array([0, 0, 0, 0, 0, 0])

for _ in range(1, 1000):    
    trial = make_input_trial()
    prediction = ISR_model_rnn.predict(trial)
    prediction = prediction[0][-(list_len+1):]
    prediction = prediction[:-1]
    prediction = prediction[newaxis,:,:]
    trial = trial[0][:-(list_len+1)]
    trial = trial[newaxis,:,:]
    for pos, (position_p, position_t) in enumerate(zip(prediction[0], trial[0])):
        if np.argmax(position_p) == np.argmax(position_t):
            position_rnn[pos] += 1
            
            
# Plots the model’s accuracy on six-item lists, 
# evaluated separately for each position.

def plot_prim_rnn():
    plt.plot(position_rnn)
    plt.xlabel('Item Position')
    plt.ylabel('Accuracy')
    plt.axis([1,6,0,1000])
    plt.show()

 

#  Word length effect

def make_input_trial_len(list_len):
    global letters_26
    trial_input = np.zeros(shape=(list_len * 2 + 1, len(letters_26) + 1))
    letters = np.random.permutation(len(letters_26))
    for i in range(list_len):
        #encoding
        trial_input[i, letters[i]] = 1
        #recall
        #recall cue
        trial_input[i + list_len, len(letters_26)] = 1
        #recall cue
    trial_input[list_len * 2, len(letters_26)] = 1
    
    return trial_input.reshape(1, list_len * 2 + 1, len(letters_26) + 1)



def examples_generator_len():
    max_list_len = 9
    while(True):
        for list_len in range(1, max_list_len + 1):
            yield make_input_trial_len(list_len)
    

len_check = examples_generator_len()

global proportion_rnn
global error_rnn
error = False
proportion_errors_rnn = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

for _ in range(900):
    trial = next(len_check)
    prediction = ISR_model_rnn.predict(trial)
    prediction = prediction[0][-(
            (
                    (len(trial[0])
                    -1)
                    //2)
                    +1):]
    prediction = prediction[:-1]
    prediction = prediction[newaxis,:,:]
    trial = trial[0][:-(
            (
                    (len(trial[0])
                    -1)
                    //2)
                    +1)]
    trial = trial[newaxis,:,:]
    pos_in_proportion = len(trial[0])-1
    
    for position_p, position_t in zip(prediction[0], trial[0]):
        if np.argmax(position_p) == np.argmax(position_t):
            error_rnn = False
            
        else:
            error_rnn = True
            break

    if error_rnn:
        proportion_errors_rnn[pos_in_proportion] += 1
            
    
def plot_w_len_rnn():
    plt.plot(proportion_errors_rnn)
    plt.xlabel('Number of items in a list')
    plt.ylabel('Errors')
    plt.axis([1,9,0,150])
    plt.show()

