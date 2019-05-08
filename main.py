import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Activation, GRU, Input, Dropout, BatchNormalization, Dense, concatenate, LSTM
import tensorflow.keras as keras
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


data = pd.read_csv('data/HY_cleaned.csv')


'''
(Currently perdicting:
1. Payment
2. Number_of_observations
3. Payment_std
4. Payment_min
5. Payment_max
6. Null probability
'''

class CONFIG:
    LOOK_WINDOW = 3
    INPUT_SHAPE = (LOOK_WINDOW,6)
    DATA_DIR = 'data/'


def prepare_data(data):
    payments = data.Payment
    d = data[['Payment','Number_of_observations','Payment_std','Payment_min','Payment_max']]
    nulls = payments.isnull()
    out = np.zeros((len(payments),6))

    for i in range(len(payments)):
        valid = [payments[i]/1e9,d.Number_of_observations[i],d.Payment_std[i]/1e9,d.Payment_min[i]/1e9,d.Payment_max[i]/1e9,1]
        non_valid = [0,0,0,0,0,0]

        if(nulls[i]):
            out[i] = non_valid
        else:
            out[i] = valid


    return out

prep_data = prepare_data(data)
prep_data.shape
prep_data = prep_data[:-1]  #getting the len to be divisible by 4 = (LOOK_WINDOW + 1)
num_samples = int(len(prep_data)/(CONFIG.LOOK_WINDOW+1))
x_train = np.zeros((num_samples,3,6))
y_train = np.zeros((num_samples,6))

for i in range(num_samples):
    x_out = []
    for j in range(CONFIG.LOOK_WINDOW):
        x_out.append(list(prep_data[4*i+j]))
    x_train[i] = np.asarray(x_out)
    y_train[i] = np.asarray(list(prep_data[4*i + CONFIG.LOOK_WINDOW]))



x_train = np.asarray(x_train,dtype=np.float32)
y_train = np.asarray(y_train,dtype=np.float32)


def CrossEntropy(yHat, y):
    loss = 0
    for pred,act in zip(yHat,y):
        if act == 1:
          loss += -tf.math.log(pred)
        else:
          loss += -tf.math.log(1 - pred)

    return loss/len(y)


def custom_loss(model,x,y):
    #assert(model.output.shape == OUTPUT_SHAPE)

    y_pred = model(x)
    nan_loss = CrossEntropy(y_pred[:,5],y[:,5])
    payments_loss = tf.reduce_mean(tf.math.square(y[:,:-1] - y_pred[:,:-1]))
    final_loss = nan_loss + payments_loss
    print("nan_loss is :" + str(nan_loss) + 'payments_loss is: ' + str(payments_loss))
    return final_loss


def compute_gradients(model,x,y):
    with tf.GradientTape() as tape:
        loss = custom_loss(model,x,y)
        gradients = tape.gradient(loss,model.trainable_variables)

    return gradients, loss

def apply_gradients(optimizer,gradients,variables):
    optimizer.apply_gradients(zip(gradients,variables))


def base_model(input_shape):

    x_input = Input(shape = input_shape)
    x = GRU(units=128,return_sequences = False)(x_input)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = GRU(units=64,return_sequences = False)(x_input)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(8,activation='relu')(x)
    x = Dense(6,activation='relu')(x)
    y_payment = Dense(5,activation='relu')(x)
    y_prob = Dense(1,activation='sigmoid')(x)

    x = concatenate([y_payment,y_prob])



    return keras.models.Model(inputs = x_input, outputs=x)


x_train[:10]

model = base_model(CONFIG.INPUT_SHAPE)

#model.summary()

epochs = 2000
learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate)

for i in range(1, epochs+1):
    gradients, loss = compute_gradients(model,x_train[:400],y_train[:400])
    apply_gradients(optimizer,gradients,model.trainable_variables)



model.save('1000epochs.h5')
model = tf.keras.models.load_model('1000epochs.h5')
#model.save('big_model_2000epochs.h5')

x_test = x_train[200:]
y_preds_test = model(x_test)
y_preds_test = np.asarray(y_preds_test)
y_preds_test[:30]
y_test = y_train[200:]

for i in range(len(y_preds_test)):
    if(y_preds_test[i,1] < 0.5):
        y_preds_test[i,:] = [0,0]

plt.plot(y_preds_test[:30,0])

plt.plot(y_test[:30,0])


plt.plot(list(zip(y_preds_test[:30,0],y_test[:30,0])))
def generator(model,seed_data,count):

    # seed_data.shape = (1,3,2)
    input = seed_data
    pred = model(input)
    i = 0
    out = []
    while(i < count):
        temp = input
        input[0,0,:] = temp[0,1,:]
        input[0,1,:] = temp[0,2,:]
        input[0,2,:] = pred[0]
        pred = model(input)
        #print('pred is: ' + str(pred))
        if(pred[0,1] < 0.5):
            out.append(0)
        else:
            out.append(float(pred[0,0]))
        i += 1
        #time.sleep(0.5)

    return out



gen = generator(model,np.expand_dims(x_train[0],0),30)


plt.plot(gen)

plt.plot(prep_data[-30:,0])
