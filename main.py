import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Activation, GRU, Input, Dropout, BatchNormalization, Dense, concatenate
import tensorflow.keras as keras


#data = pd.read_csv('data/bucket_HY_before_standarize.csv')

class CONFIG:
    INPUT_SHAPE = (3,2)
    DATA_DIR = 'data/'




def base_model(input_shape):

    x_input = Input(shape = input_shape)
    x = GRU(units=16,return_sequences = False)(x_input)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(4,activation='relu')(x)
    y_payment = Dense(1,activation=None)(x)
    y_prob = Dense(1,activation='sigmoid')(x)

    x = concatenate([y_payment,y_prob])



    return keras.models.Model(inputs = x_input, outputs=x)



model = base_model(CONFIG.INPUT_SHAPE)

model.summary()
