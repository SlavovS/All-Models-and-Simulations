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
from keras.layers import Concatenate, concatenate, Add
import keras.layers
import random
from Binding_Pool_parameters import *
from BP_STM_parameters import *
import numpy as np
from numpy import zeros, newaxis
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.utils import plot_model




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
    
        
########################
# Creating the items and the tokens
tokens_6 = [0,0,0,0,0,1]

items_26 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]


#Botvinick and Plaut model
sr_input = Input(
                shape=(None, len(tokens_6) + 1),
                name="sr_input"
            )
sr_lstm = LSTM(
            units = 50, 
            return_sequences=True,
            name="sr_lstm"
        )(sr_input)
sr_output = TimeDistributed(
            Dense(
                units=len(tokens_6) + 1,
                activation="softmax",
                name="sr_output",
            )
        )(sr_lstm)

sr_model = Model(
        inputs = sr_input, 
        outputs = [sr_output]
    )

#Binding pool model
bp_item_input = Input(shape=(None, len(items_26)), name="bp_item_input")
bp_token_input = Input(shape=(None, len(tokens_6) + 1), name="bp_token_input")
bp_all_inputs = keras.layers.concatenate([bp_item_input, bp_token_input])
bp_lstm = LSTM(
        units = 50, 
        return_sequences=True,
        name="bp_lstm"
    )(bp_all_inputs)
bp_output = TimeDistributed(
            Dense(
                units=len(items_26), 
                activation="softmax",
                name="bp_output"
            ),
        )(bp_lstm)

bp_model = Model(
        inputs = [bp_item_input, bp_token_input], 
        outputs = [bp_output]
    )



# Dual Model

sr_input_recall_cue = Input(
        shape = (None, len(tokens_6) + 1), 
        name="sr_input_recall_cue"
        )

dual_bp_all_input = keras.layers.concatenate(
        [sr_output, bp_item_input,sr_input_recall_cue]
        )
dual_lstm = LSTM(
        units = 50, 
        return_sequences=True,
        name="bp_lstml"
    )(dual_bp_all_input)
dual_output = TimeDistributed(
            Dense(
                units = len(items_26), 
                activation="softmax",
                name="bp_output"
            ),
        )(dual_lstm)

dual_model = Model(
            inputs = [sr_input, bp_item_input, sr_input_recall_cue],
            outputs = [dual_output]
        )


dual_model.load_weights("bp.weights.h5", by_name=True)
dual_model.load_weights("sr.weights.h5", by_name=True)

#plot_model(dual_model,to_file='demo.png',show_shapes=True)

#    Testing the Dual model on Simulation 1

def make_input_trial_dual():
    global tokens_6
    list_len = 6
    trial_input_token = np.zeros(shape=(list_len * 2, len(tokens_6) + 1))
    trial_input_recall_cue = np.zeros(shape=(list_len*2, len(tokens_6) + 1))
    
    letters = np.random.permutation(len(tokens_6))
    for i in range(list_len):
        #encoding
        trial_input_token[i, letters[i]] = 1

        #recall cue
        trial_input_token[i + list_len, len(tokens_6)] = 1

#    #recall cue input Puts 1 only on the first cue in the list
        trial_input_recall_cue[0, len(tokens_6)] = 1

    trial = {}
    if len(trial) == list_len:
        trial = {}
            
    global items_26
   
    trial_input_item = np.zeros(shape=(list_len, len(items_26)))
  
    for i in range(0, list_len):
        item_i = np.random.randint(0, len(items_26))
        trial_input_item[i][item_i] = 1
        
    
    return [trial_input_token.reshape(1, list_len * 2, len(tokens_6) + 1),
            trial_input_item.reshape(1, list_len, len(items_26)),
            trial_input_recall_cue.reshape(1, list_len*2, len(tokens_6) + 1)
            ]
             
        
########################                      33333333333333333333333333333333


list_len = 6
global position           
position = np.array([0, 0, 0, 0, 0, 0])

for _ in range(1, 1000):    
    trial = make_input_trial_dual()
    prediction = sr_model.predict(trial[0])
    prediction = prediction[0][-list_len:]
    prediction = prediction[newaxis,:,:]
    trial = trial[0][0][:-(list_len)]
    trial = trial[newaxis,:,:]

    for pos, (position_p, position_t) in enumerate(zip(prediction[0], trial[0][0])):
        if np.argmax(position_p) == np.argmax(position_t):
            position[pos] += 1
            
print(position)

def plot_prim_lstm():
    plt.plot(position)
    plt.xlabel('Item Position')
    plt.ylabel('Accuracy')
    plt.axis([1,6,0,1000])
    plt.show()
print(plot_prim_lstm())
