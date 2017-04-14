from riksmot import DATA_PATH

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

import tensorflow as tf
import keras
tf.python.control_flow_ops = tf

from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.layers import Input
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM, Input

SEQ_LEN=10
MAX_FILES=10

unique_chars = set()
for filename in os.listdir(DATA_PATH)[:MAX_FILES]:
    with open(os.path.join(DATA_PATH, filename), 'r') as f:
        unique_chars |= set(f.read().lower())

char_to_int = {c: i for i, c in enumerate(unique_chars)}
int_to_char = {v: k for k,v in char_to_int.items()}

def get_data():

    X = []
    y = []
    for filename in os.listdir(DATA_PATH)[:MAX_FILES]:

        with open(os.path.join(DATA_PATH, filename), 'r') as f:
            text = f.read().lower()
        digits = [char_to_int[c] for c in text if ord(c)]

        for i in range(0, len(digits) - SEQ_LEN):
            X.append(digits[i:i+SEQ_LEN])
            y.append(digits[i+SEQ_LEN])
    X = np.array(X)
    y = np.array(y)
    return X, y


def build_rnn(input_shape, n_classes):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    #model.add(LSTM(32, return_sequences=True))
    #model.add(LSTM(32))

    model.add(Dense(n_classes, activation='softmax'))

    return model

def gen_text(model, seed, mu, sigma):
    length = 100
    ints = np.zeros(length, dtype=seed.dtype)
    ints[:len(seed)] = np.squeeze(seed)

    for i in range(0, length - SEQ_LEN):
        x = ints[i:i+SEQ_LEN].reshape((1, -1, 1)).astype(np.float)
        x -= mu
        x /= sigma
        pred = model.predict(x)
        ints[i + SEQ_LEN] = np.argmax(pred)
    return "".join(int_to_char[i] for i in ints)

def main():

    epochs = 10

    X, y = get_data()
    n_seeds = 5
    seeds = X[np.random.randint(0, len(X), n_seeds)]

    X = np.expand_dims(X, -1).astype(np.float)
    Y = np_utils.to_categorical(y)

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)

    X -= mu
    X /= sigma

    model = build_rnn(X[0].shape, Y.shape[1])

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(X, Y, nb_epoch=epochs, batch_size=128, verbose=2)


    for seed in seeds:
        print(gen_text(model, seed, mu, sigma))

if __name__ == '__main__':
    main()
    import gc
    gc.collect()
