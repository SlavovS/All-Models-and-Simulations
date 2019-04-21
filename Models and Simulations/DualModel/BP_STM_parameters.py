# Parameters for Botvinick & Plaut model


length_of_list = 6
training_examples = 70
test_examples = 30

letters_3 = [0,0,1]

letters_26 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]


cycles = 1000  # Cycle = passing through the full range of list lengths
             # Trial = processing of a single list (encoding + recall)
             
             
n_features = 26

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
from keras import backend as K

tokens_6 = [0,0,0,0,0,1]