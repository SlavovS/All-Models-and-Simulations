
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
from keras.layers import Concatenate
import keras.layers

# Saving the training status
csv_logger = CSVLogger('status_log.csv', append=True, separator=',')

#  STM model for serial recall   (Botvinick & Plaut, 2006)
  
    
from BP_STM_parameters import *



# Generating the input and output for training


def make_trial(list_len):
    global tokens_6
    trial_input = np.zeros(shape=(list_len * 2, len(tokens_6) + 1))
    trial_input_recall_cue = np.zeros(shape=(list_len*2, len(tokens_6) + 1))
    
    trial_output = np.zeros(shape=(list_len * 2, len(tokens_6) + 1))
    letters = np.random.permutation(len(tokens_6))
    for i in range(list_len):
        #encoding
        trial_input[i, letters[i]] = 1
        trial_output[i, letters[i]] = 1
        #recall
        #recall cue
        trial_input[i + list_len, len(tokens_6)] = 1
        #output letter
        trial_output[i + list_len, letters[i]] = 1
#    #recall cue input Puts 1 only on the first cue in the list
        trial_input_recall_cue[0, len(tokens_6)] = 1
        
#    trial_input_recall_cue[0][len(letters_26)] = 1
#    trial_input[list_len * 2, len(letters_26)] = 1
#    #end of list
#    trial_output[list_len * 2, len(letters_26)] = 1
#    
    return (
            [trial_input.reshape(1, list_len * 2, len(tokens_6) + 1),
             trial_input_recall_cue.reshape(1, list_len*2, len(tokens_6) + 1)],
            trial_output.reshape(1, list_len *2, len(tokens_6) + 1)
            )
    

    
def examples_generator():
    max_list_len = 6
    while(True):
        for list_len in range(1, max_list_len + 1):
            yield make_trial(list_len)
    

#############################################################
# Test set
def make_trial_val(list_len):
    global tokens_6
    trial_input_val = np.zeros(shape=(list_len * 2, len(tokens_6) + 1))
    trial_input_recall_cue_val = np.zeros(shape=(list_len*2, len(tokens_6) + 1))
    
    trial_output_val = np.zeros(shape=(list_len * 2, len(tokens_6) + 1))
    letters = np.random.permutation(len(tokens_6))
    for i in range(list_len):
        #encoding
        trial_input_val[i, letters[i]] = 1
        trial_output_val[i, letters[i]] = 1
        #recall
        #recall cue
        trial_input_val[i + list_len, len(tokens_6)] = 1
        #output letter
        trial_output_val[i + list_len, letters[i]] = 1
    
    #recall cue Represented as a separate input for the Dual model to work
    ##trial_input_val[list_len * 2, len(letters_26)] = 1
        trial_input_recall_cue_val[0, len(tokens_6)] = 1
    
        
    #end of list No end of list for the final Dual Model to work
    #trial_output_val[list_len * 2, len(letters_26)] = 1
    
    return (
            [trial_input_val.reshape(1, list_len * 2, len(tokens_6) + 1),
             trial_input_recall_cue_val.reshape(1, list_len*2, len(tokens_6) + 1)],
            trial_output_val.reshape(1, list_len *2, len(tokens_6) + 1)
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
            

            # Building the Model

sr_input = Input(
                shape=(None, len(tokens_6) + 1),
                name="sr_input"
            )
sr_input_recall_cue = Input(
        shape = (None, len(tokens_6) + 1), 
        name="sr_input_recall_cue"
        )
sr_all_inputs = keras.layers.add([sr_input, sr_input_recall_cue])
sr_lstm = LSTM(
            units = 50, 
            return_sequences=True,
            name="sr_lstm"
        )(sr_all_inputs)
sr_output = TimeDistributed(
            Dense(
                units=len(tokens_6) + 1,
                activation="softmax",
                name="sr_output",
            )
        )(sr_lstm)

sr_model = Model(
        inputs = [sr_input, sr_input_recall_cue], 
        outputs = [sr_output]
    )

         
         
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
sr_model.compile(loss="categorical_crossentropy", 
                 optimizer= Adam(lr = 0.001), metrics=['accuracy', my_accuracy])

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
es =  EarlyStoppingByAccuracy(monitor='val_my_accuracy', value=0.58, verbose=1)
mc = ModelCheckpoint('best_model_sr_CHERNOVA.h5', monitor='val_my_accuracy', mode='max', verbose=1,
                     save_best_only=True)

history = sr_model.fit_generator(        
        examples_generator(),
#        batch_size=1, 
        nb_epoch=10000,
        steps_per_epoch = 1000,
        verbose=2, 
        validation_data = examples_generator_val(),
        nb_val_samples = 300,
        
        callbacks = [es, mc, csv_logger]
    )


sr_model.save('sr_model.h5') 
sr_model.save_weights("sr.weights.h5")


# plot training history
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
#plt.legend()
#plt.show()



from keras.models import load_model
ISR_model = load_model('sr_model.h5')




