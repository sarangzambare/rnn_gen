import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Activation, GRU, Input, Dropout, BatchNormalization, Dense, concatenate, LSTM
import tensorflow.keras as keras
import numpy as np


data = pd.read_csv('data/bucket_HY_before_standarize.csv')

class CONFIG:
    LOOK_WINDOW = 3
    INPUT_SHAPE = (LOOK_WINDOW,2)
    DATA_DIR = 'data/'


def prepare_data(data):
    payments = data.Payment
    nulls = payments.isnull()
    out = np.zeros((len(payments),2))

    for i in range(len(payments)):
        valid = [payments[i]/1e5,1]
        non_valid = [0,0]

        if(nulls[i]):
            out[i] = non_valid
        else:
            out[i] = valid


    return out

prep_data = prepare_data(data)
prep_data = prep_data[:-1]  #getting the len to be divisible by 4 = (LOOK_WINDOW + 1)

num_samples = int(len(prep_data)/(CONFIG.LOOK_WINDOW+1))
x_train = np.zeros((num_samples,3,2))
y_train = np.zeros((num_samples,2))

for i in range(num_samples):
    x_out = []
    for j in range(CONFIG.LOOK_WINDOW):
        x_out.append(list(prep_data[4*i+j]))
    x_train[i] = np.asarray(x_out)
    y_train[i] = np.asarray(list(prep_data[4*i + CONFIG.LOOK_WINDOW]))



x_train = np.asarray(x_train,dtype=np.float32)
y_train = np.asarray(y_train,dtype=np.float32)





def custom_loss(model,x,y):
    #assert(model.output.shape == OUTPUT_SHAPE)

    y_pred = model(x)

    loss = tf.reduce_mean(tf.math.square(y[0] - y_pred[0]) + tf.math.square(y[1]-y_pred[1]))
    print("loss is :" + str(loss))
    return loss


def compute_gradients(model,x,y):
    with tf.GradientTape() as tape:
        loss = custom_loss(model,x,y)
        gradients = tape.gradient(loss,model.trainable_variables)

    return gradients, loss

def apply_gradients(optimizer,gradients,variables):
    optimizer.apply_gradients(zip(gradients,variables))


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

#model.summary()


epochs = 1000
learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate)

for i in range(1, epochs+1):
    gradients, loss = compute_gradients(model,x_train,y_train)
    apply_gradients(optimizer,gradients,model.trainable_variables)
