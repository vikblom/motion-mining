from riksmot import DATA_PATH, NET_PATH

import signal
import sys
import numpy as np
from os import listdir
from os.path import join

import tensorflow as tf
import keras

from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.callbacks import Callback
from keras.layers import Dense, LSTM


SEQ_LEN = 15
STEP = 1
FILES_PER_BATCH = 5

chars = [chr(i) for i in range(ord('a'), ord('z')+1)] + \
        ['å', 'ä', 'ö', '.', ',', ' ']#, '(', ')', '\n']
n_chars = len(chars)

char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {v: k for k,v in char_to_int.items()}


def make_dir_batcher(sub_dir):
    """
    Returns a generator and the number of files in
    the given sub-directory, i.e. "items in epoch".
    The generator is infinite and returns X and Y
    for one file at a time.
    """
    # Prepare a shuffle
    filenames = listdir(join(DATA_PATH, sub_dir))
    indexes = np.arange(len(filenames))
    def dir_batcher():
        while True:
            np.random.shuffle(indexes)
            for filename in (filenames[i] for i in indexes):
                file_path = join(DATA_PATH, sub_dir, filename)
                yield data_from_file(file_path)
    return dir_batcher(), len(filenames)


def data_from_file(file_path):
    X = []
    y = []
    with open(file_path, 'r') as f:
        text = f.read().lower()
    digits = [char_to_int[c] for c in text if c in chars]

    X, y = [], []
    for i in range(0, len(digits) - SEQ_LEN, STEP):
        X.append(digits[i:i+SEQ_LEN])
        y.append(digits[i+SEQ_LEN])

    X = np.array(X)
    y = np.array(y)

    return to_one_hot(X), to_one_hot(y)


def to_one_hot(data):
    shape = data.shape
    data = np_utils.to_categorical(data, nb_classes=n_chars)
    return np.reshape(data, shape + (n_chars,))


def build_rnn(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True,
                   input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))

    model.add(Dense(n_chars, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop')

    return model


class Writer(Callback):

    def __init__(self, seed):
        self.seed = seed

    def on_epoch_end(self, epoch, logs={}):
        print('\nSeed:{}|'.format("-"*(SEQ_LEN-6)))
        print(gen_text(self.model, self.seed))
        print()
        return


def gen_text(model, seed):
    length = 1000
    ints = np.zeros(length, dtype=seed.dtype)
    ints[:len(seed)] = np.argmax(seed, axis=1)

    seed = np.expand_dims(seed, 0)
    for i in range(SEQ_LEN, length):

        pred = model.predict(seed)
        choice = np.random.multinomial(1, pred[-1], 1)
        ints[i] = np.argmax(choice)
        seed = np.vstack((seed, [np.r_[seed[-1][1:], choice]]))

    return "".join(int_to_char[i] for i in ints)


def main():

    X, Y = data_from_file(join(DATA_PATH, "strindberg.txt"))

    seed_text = "det var en gång"
    seed = to_one_hot(np.array([char_to_int[c] for c in seed_text]))
    #seed = X[np.random.randint(0, len(X))]

    model = build_rnn(X[0].shape)

    epoch = 0
    while keep_going:
        model.fit(X, Y, nb_epoch=epoch+1, batch_size=1024,
                  callbacks=[Writer(seed)],
                  initial_epoch = epoch,
                  shuffle=False, verbose=1)
        epoch += 1


    model.save(join(NET_PATH,
                    "rnn_seq_{}.hdf5".format(SEQ_LEN)))

    return model


def stop_training(signum, frame):
    # Restore original SIGINTE so that further C-c does their job.
    signal.signal(signal.SIGINT, original_sigint)

    global keep_going
    keep_going = False


if __name__ == '__main__':
    keep_going = True

    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, stop_training)

    model = main()
