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
from keras.layers import Concatenate
import keras.layers

import random
from Binding_Pool_parameters import *
from BP_STM_parameters import *
import numpy as np
from numpy import zeros, newaxis
from keras.models import load_model
import matplotlib.pyplot as plt
        
########################
# Creating the items and the tokens
tokens_6 = [0,0,0,0,0,1]

items_26 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]


# Generate random token
token_perm = []
def gen_token():
    global token_perm
    
    if len(token_perm) == 0:
        token_perm = np.random.permutation(len(tokens_6))
        
    token = token_perm[0]
    token_perm = token_perm[1:]
    
    return token

# Training
def make_trial(list_len):
    trial = {}
    if len(trial) == list_len:
        trial = {}
            
    global items_26
    #list_len = len(tokens_6)
    trial_input_token = np.zeros(shape=(list_len, len(tokens_6) + 1))
    trial_input_item = np.zeros(shape=(list_len, len(items_26)))
    trial_output_item = np.zeros(shape=(list_len, len(items_26)))
    #letters = np.random.permutation(len(items_26))
    token_i = np.random.randint(0, len(tokens_6)) # Index of the token 
    item_i = np.random.randint(0, len(items_26))
    trial_input_item[0][item_i] = 1
    trial_output_item[0][item_i] = 1
    trial_input_token[0][token_i] = 1

    trial[token_i] = item_i
    for i in range(1, list_len):
        if np.random.rand() < 0.5:
            # Recall
            rand_token = random.choice(list(trial.keys()))
            trial_output_item[i, trial[rand_token]] = 1
            trial_input_token[i, rand_token] = 1
            trial_input_token[i, len(tokens_6)] = 1 # This is the recall cue
                                                    # for this specific token
            
            
        else:
            # Encoding
            
            token_i = gen_token()
            item_i = np.random.randint(0, len(items_26))
            trial_input_item[i][item_i] = 1
            trial_output_item[i][item_i] = 1
            trial_input_token[i][token_i] = 1
            trial_input_token[i, len(tokens_6)] = 0 # Encoding cue
            trial[token_i] = item_i
        
    return (    
            [trial_input_item.reshape(1, list_len, len(items_26)), 
            trial_input_token.reshape(1, list_len, len(tokens_6) + 1)],
            trial_output_item.reshape(1, list_len, len(items_26))
            )

def examples_generator():
    while(True):
        yield make_trial(20)


######################################
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
    def __init__(self, monitor='val_acc', value=0.98, verbose=0):
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




####################################################################
# Creating the network layer by layer


bp_item_input = Input(shape=(None, len(items_26)), name="bp_item_input")
bp_token_input = Input(shape=(None, len(tokens_6) + 1), name="bp_token_input")
bp_all_inputs = keras.layers.concatenate([bp_item_input, bp_token_input])
bp_lstm = LSTM(
        units = 30, 
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



#  Fitting

bp_model.compile(loss="categorical_crossentropy", 
                 optimizer= Adam(lr = 0.01), metrics=['accuracy', my_accuracy])

es =  EarlyStoppingByAccuracy(monitor='val_my_accuracy', value=0.98, verbose=1)
mc = ModelCheckpoint('best_model_bp_intermediate.h5', monitor='val_my_accuracy', mode='max', verbose=1,
                     save_best_only=True)

history = bp_model.fit_generator(        
        examples_generator(),
#        batch_size=1, 
        nb_epoch=30,
        steps_per_epoch = 1000,
        verbose=1, 
        validation_data = examples_generator(),
        nb_val_samples = 300,
        
        callbacks = [es,mc]
    )
bp_model.save("Binding_Pool_model_30_my_acc.h5")
bp_model.save_weights("bp.weights_30_my_acc.h5")

bp_model = load_model('best_model_bp_30.h5')
#  Testing the Binding pool model
#
def check_bp():
    global accuracy_bp
    global accuracy_bp_1
    global total
    total = 0
    accuracy_bp = []
    accuracy_bp_1 = 0
    
    for _ in range(1,100):
        accuracy_bp_step = 0
        test = make_trial(6)
        prediction_bp = bp_model.predict(test[0])
         
        for pred, real in zip(prediction_bp[0], test[0][0][0]):
            total += 1
            if np.argmax(pred)==np.argmax(real):
                accuracy_bp_step += 1
                accuracy_bp_1 += 1
                accuracy_bp.append(accuracy_bp_step)
        
    return accuracy_bp_1*100/total
    
        