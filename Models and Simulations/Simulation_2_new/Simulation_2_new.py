# Simulation 2 (New)

# Testing with only 1 letter not being trained on a single position

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

    
from BP_STM_parameters import *


   
# Generating the input and output for training


def make_trial(list_len):
    global letters_26
    trial_input = np.zeros(shape=(list_len * 2 + 1, len(letters_26) + 1))
    trial_output = np.zeros(shape=(list_len * 2 + 1, len(letters_26) + 1))
    letters = np.random.permutation(len(letters_26))
    for i in range(list_len):
        
        # Excluding letter A from position 5 (Condition 5)
        if letters[4] == 0:
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
            
            
            # Test set
def make_trial_val(list_len):
    global letters_26
    trial_input_val = np.zeros(shape=(list_len * 2 + 1, len(letters_26) + 1))
    trial_output_val = np.zeros(shape=(list_len * 2 + 1, len(letters_26) + 1))
    letters = np.random.permutation(len(letters_26))
    for i in range(list_len):
        if letters[4] == 0:
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
         
         
BP_model = Sequential()
#BP_model = load_model('Ready_model_rnn_1.h5', custom_objects={'my_accuracy': my_accuracy})
BP_model.add(
        LSTM(
                units = 50, 
                input_shape=(None, len(letters_26) + 1), 
                return_sequences=True
        )
    )
    
BP_model.add(
        TimeDistributed(
                Dense(
                        units = len(letters_26) + 1,
                        activation="softmax"
                )
            )
        )
BP_model.compile(loss="categorical_crossentropy", 
                 optimizer= Adam(lr = 0.001), metrics=['accuracy', my_accuracy])

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
es =  EarlyStoppingByAccuracy(monitor='val_my_accuracy', value=0.58, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_my_accuracy', mode='max', verbose=1,
                     save_best_only=True)

history = BP_model.fit_generator(        
        examples_generator(),
#        batch_size=1, 
        nb_epoch=10000,
        steps_per_epoch = 1000,
        verbose=2, 
        validation_data = examples_generator_val(),
        nb_val_samples = 300,
        
        callbacks = [es, mc, csv_logger]
    )
#BP_model.save('Ready_model_rnn_2.h5') 
BP_model.save('sr_model_Condition_5.h5') 


#  Test 
      
            # make lists with only A s on the 6 positions
sr_model = load_model(
        'sr_model_Condition_1.h5', custom_objects={'my_accuracy': my_accuracy}
        )
#Ready_BP_model.h5
#sr_model_Condition_1.h5'
list_len = 6
def make_input_trial():
    global letters_26
    list_len = 6
    trial_input = np.zeros(shape=(list_len * 2 + 1, len(letters_26) + 1))
    
    # Creating letter A on all 6 positions
    letters = np.full(len(letters_26), 0)
    for i in range(list_len):        
        
       
        #encoding
        trial_input[i, letters[i]] = 1
        #recall
        #recall cue
        trial_input[i + list_len, len(letters_26)] = 1
        #recall cue
    trial_input[list_len * 2, len(letters_26)] = 1
    
    return trial_input.reshape(1, list_len * 2 + 1, len(letters_26) + 1)

# Run the simulation 1000 times and plot the accuracy of letter A on 
# different posiions

global position 
position = np.zeros([6,10000])
passes = 0


for i in range(0, 10000):    
    trial = make_input_trial()
    if trial is None:
        passes+=1
        pass
    
    else:
        prediction = sr_model.predict(trial)
        prediction = prediction[0][-(list_len+1):]
        prediction = prediction[:-1]
        prediction = prediction[newaxis,:,:]
        trial = trial[0][:-(list_len+1)]
        trial = trial[newaxis,:,:]
    
        for pos, (position_p, position_t) in enumerate(zip(prediction[0], trial[0])):
            if np.argmax(position_p) == np.argmax(position_t):
                position[pos][i] += 1

print(position)   

position_1_mean = np.mean(position[0])
position_2_mean = np.mean(position[1])
position_3_mean = np.mean(position[2])
position_4_mean = np.mean(position[3])
position_5_mean = np.mean(position[4])
position_6_mean = np.mean(position[5])

position_1_sd = np.std(position[0])
position_2_sd = np.std(position[1])
position_3_sd = np.std(position[2])
position_4_sd = np.std(position[3])
position_5_sd = np.std(position[4])
position_6_sd = np.std(position[5])
    
