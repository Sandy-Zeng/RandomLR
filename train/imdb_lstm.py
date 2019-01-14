'''Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
# Notes
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np
import time
from keras.callbacks import LearningRateScheduler
import sys
from pathlib import *
from keras.optimizers import SGD

def U(tmp_lr,random_range):
    np.random.seed(int(time.time()))
    tmp_lr = np.random.random() * tmp_lr * random_range
    return tmp_lr

init_lr = 1e-1
lr_method = sys.argv[1]
random_range = int(sys.argv[2])
work_path = Path('/home/ouyangzhihao/Backup/Exp/ZYY/RandomLR/Final/DataSets/res.txt')
epochs = 15

exp_name = 'IMDB_%s_%d'%(lr_method,random_range)

def lr_schedule(epoch):
    lr = init_lr
    if epoch > epochs*0.75:
        lr = init_lr * 1e-2
    elif epoch > epochs*0.5:
        lr = init_lr * 1e-1
    if lr_method == 'U' and epoch>epochs*0.5:
        lr = U(lr,random_range)
    print('Learning Rate:',lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)
callbacks = [lr_scheduler]

max_features = 20000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer= SGD(lr=init_lr,momentum=0.9),
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks=callbacks)
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)