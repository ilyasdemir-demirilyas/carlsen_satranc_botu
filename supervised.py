'''
burada keras modelinin doğruluk metriği olarak almadığı doğrulama metrikleri bulunuyor .
ters durumunlarda modelin doğruluğunu kontrol etmek için kullanılıyor .
'''


from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

'''
Burada denetimli öğrenme ile keras modeli oluşturulur .
'''

import tensorflow.compat.v2 as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules
import numpy as np
import os
from keras.utils import Sequence
from keras import backend as B
from keras.layers import Dense, Input, Concatenate
from keras.models import Model, load_model
import keras


supervised_layers = [700, 500, 250, 2]


DBNpath = "./network/DBN.hdf5"

class Data(Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size

        positions = np.load("./data/positions.npy")
        results = np.load('./data/results.npy')

        self.white_positions = positions[results == 1]
        self.black_positions = positions[results == 0]

        self.white_positions = self.white_positions[:len(self.black_positions)]


        np.random.shuffle(self.white_positions)
        np.random.shuffle(self.black_positions)

        print("good so far!")

    def __len__(self):
        return int(np.ceil(len(self.black_positions)/float(self.batch_size)))

    def __getitem__(self, index):
        startIndex = index * self.batch_size

        if (startIndex + self.batch_size < len(self.black_positions)):
            w_batch = self.white_positions[startIndex : startIndex + self.batch_size]
            b_batch = self.black_positions[startIndex : startIndex + self.batch_size]
        else:
            w_batch = self.white_positions[startIndex:]
            b_batch = self.black_positions[startIndex:]


        w_results = np.ones((len(w_batch),))
        b_results = np.zeros((len(b_batch),))

        X = np.stack([w_batch, b_batch], axis = 1)

        outputs = np.stack([w_results, b_results], axis = 1)

        randomization = np.random.randint(2, size = len(X))

        X[randomization == 1] = np.flip(X[randomization == 1], axis = 1)

        outputs[randomization == 1] = np.flip(outputs[randomization == 1], axis = 1)

        batch1, batch2 = np.split(X, 2, axis = 1)

        batch1 = np.squeeze(batch1)

        batch2 = np.squeeze(batch2)

        return [batch1, batch2], outputs


    def on_epoch_end(self):
        print("epoch over")
        np.random.shuffle(self.white_positions)
        np.random.shuffle(self.black_positions)

data_object = Data(256)

print("data object initialized")

DBN = load_model(DBNpath)

top_input = Input(shape = (774,))
bottom_input = Input(shape = (774,))


DBN_top_out = DBN(top_input)
DBN_bottom_out = DBN(bottom_input)

complete_input = Concatenate()([DBN_top_out,DBN_bottom_out])


layer1 = Dense(supervised_layers[0], activation='relu')(complete_input)
layer2 = Dense(supervised_layers[1], activation='relu')(layer1)
layer3 = Dense(supervised_layers[2], activation='relu')(layer2)
output_layer = Dense(supervised_layers[3], activation='softmax')(layer3)


pooya_and_associates_model = Model([top_input, bottom_input], output_layer)
rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.98)
opt = keras.optimizers.Adam(learning_rate=rate)

pooya_and_associates_model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['acc'])

history=pooya_and_associates_model.fit_generator(data_object, epochs = 3000)





pooya_and_associates_model.save("./network/DeepLearningModel.h5")

'''
en son değerler .
Epoch 2999/3000
234/234 [==============================] - 1s 6ms/step - loss: 0.1823 - acc: 0.9153
epoch over
Epoch 3000/3000
234/234 [==============================] - 1s 6ms/step - loss: 0.1814 - acc: 0.9165
'''

'''3020     8610
oluşan loss ve acc değerlerinin görseleştirmesi burada yapılır .
hangi adımda hangi değer aldıkları grafikte görselleştirilir
'''

import pandas as pd 
import matplotlib.pyplot as plt


model_history = pd.DataFrame(history.history)
model_history['epoch'] = history.epoch


fig, ax = plt.subplots(1, figsize=(5,5))
num_epochs = model_history.shape[0]

ax.plot(np.arange(0, num_epochs), model_history["loss"], 
        label="loss",color="orange")
ax.legend()
plt.tight_layout()




ax.plot(np.arange(0, num_epochs), model_history["acc"], 
        label="acc",color="red")
ax.legend()
plt.tight_layout()


plt.show()

