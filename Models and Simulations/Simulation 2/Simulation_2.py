#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 08:31:14 2019

@author: slavi
"""
# Simulation 2 --> Replicating not performing well when not trained on specific position
# Bowers et al., 2009

## we tested it on sequences in which some letters were excluded
#from specific positions. We trained the Botvinick and Plaut model
#in the same way as above except that we ensured that four letters
#never occurred in specific positions, namely, 
# Cond_1 B-in-Position-1, = index 1. 1000 trials
# Cond_2 D-in-Position-2, = index 3
# Cond_3 G-in-Position-3, = index 6 and 
# Cond_4 J-in-Position-4. = index 9  Cond_4 should be all 4 letters
#  I will repeat only 1 and 4

# When a random list was generated in which one (or more) of these letters
#occurred in these positions, we simply eliminated that list and
#generated another sequence. At recall we tested the model on lists
#of 1,000 six-letter sequences when zero (baseline), one, two, three,
#or all four of the critical untrained letter–position combinations
#were included in the list. 


####################################################################
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Input
from keras import backend as K
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, Callback, warnings
import matplotlib.pyplot as plt
from keras.optimizers import Adam

# Saving the training status
csv_logger = CSVLogger('status_log.csv', append=True, separator=',')

#  STM model for serial recall   (Botvinick & Plaut, 2006)
  
    
from BP_STM_parameters import *


# Generating the input and output for training


def make_trial(list_len):
    global letters_26
    trial_input = np.zeros(shape=(list_len * 2 + 1, len(letters_26) + 1))
    trial_output = np.zeros(shape=(list_len * 2 + 1, len(letters_26) + 1))
    letters = np.random.permutation(len(letters_26))
    for i in range(list_len):
        
        # Excluding B, D, G, J from positions 1, 2, 3, and 4, respectively
        if letters[0] == 1:
            pass
        if letters[1] == 3:
            pass
        if letters[2] == 6:
            pass
        if letters[3] == 9:
            pass
        else:
            #encoding
            trial_input[i, letters[i]] = 1
            trial_output[i, letters[i]] = 1
            #recall
            #recall cue
            trial_input[i + list_len, len(letters_26)] = 1
            #output letter
            trial_output[i + list_len, letters[i]] = 1
        #recall cue
        trial_input[list_len * 2, len(letters_26)] = 1
        #end of list
        trial_output[list_len * 2, len(letters_26)] = 1
        
    return (
            trial_input.reshape(1, list_len * 2 + 1, len(letters_26) + 1), 
            trial_output.reshape(1, list_len *2 + 1, len(letters_26) + 1)
            )


    
def examples_generator():
    max_list_len = 9
    while(True):
        for list_len in range(1, max_list_len + 1):
            yield make_trial(list_len)
   

#############################################################
# Test set
def make_trial_val(list_len):
    global letters_26
    trial_input_val = np.zeros(shape=(list_len * 2 + 1, len(letters_26) + 1))
    trial_output_val = np.zeros(shape=(list_len * 2 + 1, len(letters_26) + 1))
    letters = np.random.permutation(len(letters_26))
    for i in range(list_len):
        if letters[0] == 1:
            pass
        if letters[1] == 3:
            pass
        if letters[2] == 6:
            pass
        if letters[3] == 9:
            pass
        else:
            #encoding
        
            trial_input_val[i, letters[i]] = 1
            trial_output_val[i, letters[i]] = 1
            #recall
            #recall cue
            trial_input_val[i + list_len, len(letters_26)] = 1
            #output letter
            trial_output_val[i + list_len, letters[i]] = 1
        #recall cue
        trial_input_val[list_len * 2, len(letters_26)] = 1
        #end of list
        trial_output_val[list_len * 2, len(letters_26)] = 1
        
    return (
            trial_input_val.reshape(1, list_len * 2 + 1, len(letters_26) + 1), 
            trial_output_val.reshape(1, list_len *2 + 1, len(letters_26) + 1)
            )

inputs_val = []
outputs_val = []

# Validation steps should typically be equal to the number of samples of 
# your validation dataset divided by the batch size
    
def examples_generator_val():
#    max_list_len = 9
    while(True):
#        for list_len in range(1, max_list_len + 1):
         yield make_trial_val(6)




    
# This acc function takes into account the output during recall only
def my_accuracy(y_true, y_pred):
#    print(y_pred)
    return K.min(
                K.cast(
                    K.equal(
                            K.argmax(y_true, axis = 2), 
                            K.argmax(y_pred, axis = 2)
                    ), dtype="float32"
                )
            )
    


##############################################################
#

# Stopping training after specific validation accuracy is reached
class EarlyStoppingByAccuracy(Callback):
    def __init__(self, monitor='val_my_accuracy', value=0.58, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            return warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current >= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
##############################################################

#  Fitting the model
         
#         
#BP_model = Sequential()
##BP_model = load_model('Ready_model_rnn_1.h5', custom_objects={'my_accuracy': my_accuracy})
#BP_model.add(
#        LSTM(
#                units = 50, 
#                input_shape=(None, len(letters_26) + 1), 
#                return_sequences=True
#        )
#    )
#    
#BP_model.add(
#        TimeDistributed(
#                Dense(
#                        units = len(letters_26) + 1,
#                        activation="softmax"
#                )
#            )
#        )
#BP_model.compile(loss="categorical_crossentropy", 
#                 optimizer= Adam(lr = 0.001), metrics=['accuracy', my_accuracy])
#
##es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
#es =  EarlyStoppingByAccuracy(monitor='val_my_accuracy', value=0.58, verbose=1)
#mc = ModelCheckpoint('best_model.h5', monitor='val_my_accuracy', mode='max', verbose=1,
#                     save_best_only=True)
#
#history = BP_model.fit_generator(        
#        examples_generator(),
##        batch_size=1, 
#        nb_epoch=10000,
#        steps_per_epoch = 1000,
#        verbose=2, 
#        validation_data = examples_generator_val(),
#        nb_val_samples = 300,
#        
#        callbacks = [es, mc, csv_logger]
#    )
##BP_model.save('Ready_model_rnn_2.h5') 
#BP_model.save('BP_model_Cond_4.h5') 


####################################################################

#   This is the key simulation that I have to repeat with the Combined model
            # and I would expect not to be 4% accuracy


#  Test 
# at test, we included 1,000 trials in which
#B-in-Position-1 was tested, 1,000 trials in which D-in-Position-2
#was presented, etc. Similarly, when two critical untrained letters
#were presented at test, we included 1,000 trials in which B-inPosition-1 and D-in-Position-2 were tested, 1,000 trials in which
#D-in-Position-2 and G-in-Position-3 were tested, etc. Other than
#this restriction, the sequence of the six letters in the lists was
#random at test. T

ISR_model = load_model(
        'BP_model_Cond_4.h5', custom_objects={'my_accuracy': my_accuracy}
        )

list_len = 6
def make_input_trial():
    global letters_26
    list_len = 6
    trial_input = np.zeros(shape=(list_len * 2 + 1, len(letters_26) + 1))
    letters = np.random.permutation(len(letters_26))
    for i in range(list_len):
        
       
        if letters[0] != 1:
            pass
        if letters[1] != 3:
            pass
        if letters[2] != 6:
            pass
        if letters[3] != 9:
            pass
        else:
        #encoding
            trial_input[i, letters[i]] = 1
        #recall
        #recall cue
            trial_input[i + list_len, len(letters_26)] = 1
        #recall cue
    trial_input[list_len * 2, len(letters_26)] = 1
    
    return trial_input.reshape(1, list_len * 2 + 1, len(letters_26) + 1)



#def make_input_trial():
#    global letters_26
#    list_len = 6
#    trial_input = np.zeros(shape=(list_len * 2 + 1, len(letters_26) + 1))
#    letters = np.random.permutation(len(letters_26))
#    for i in range(list_len):
#    trial_input[i, letters[i]] = 1
#            #recall
#            #recall cue
#            trial_input[i + list_len, len(letters_26)] = 1
#            #recall cue
#        trial_input[list_len * 2, len(letters_26)] = 1
#        
#    return trial_input.reshape(1, list_len * 2 + 1, len(letters_26) + 1)
# 


global position           
position = np.array([0, 0, 0, 0, 0, 0])

for _ in range(1, 10000):    
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
            
            
# Plots the model’s accuracy on six-item lists, 4% accuracy
# evaluated separately for each position.
def plot_prim_lstm():
    plt.plot(position)
    plt.xlabel('Item Position')
    plt.ylabel('Accuracy')
    plt.axis([1,6,0,10000])
    plt.show()
    
plot_prim_lstm()
