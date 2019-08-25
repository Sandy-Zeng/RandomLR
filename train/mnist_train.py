'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import time
from keras.callbacks import LearningRateScheduler
import sys
from pathlib import *
from keras.utils.vis_utils import plot_model
import sys
sys.path.append('/home/zengyuyuan/RandomLR')
from scheduler.optimizers import SGD_RandomGradient
from keras.callbacks import TensorBoard

batch_size = 128
num_classes= 10
epochs = 60

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

init_lr = float(sys.argv[5])
lr_method = sys.argv[1]
random_range = int(sys.argv[2])
work_path = Path('/home/zengyuyuan/RandomLR/')
fcn_depth = 17
sgd = sys.argv[3]
log_path = sys.argv[4]

exp_name = 'MNIST_%s_%d_fcn%d_%s_epoch%d_initlr_%.2f'%(lr_method,random_range,fcn_depth,sgd,epochs,init_lr)

def U(tmp_lr,random_range):
    np.random.seed(int(time.time()))
    tmp_lr = np.random.random() * tmp_lr * random_range
    return tmp_lr

def lr_schedule(epoch):
    lr = init_lr
    # if epoch > epochs*0.75:
    #     lr = init_lr * 1e-2
    # elif epoch > epochs*0.5:
    #     lr = init_lr * 1e-1
    if lr_method == 'U' and epoch>epochs*0.8:
        lr = U(lr,random_range)
    print('Learning Rate:',lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)
TB_log_path = '/home/zengyuyuan/RandomLR/' + log_path + exp_name
callbacks = [TensorBoard(log_dir=(TB_log_path.__str__()))]
if lr_method == 'U':
    callbacks.append(lr_scheduler)


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# x_train = x_train.reshape(x_train.shape[0],img_rows*img_cols)
# x_test = x_test.reshape(x_test.shape[0],img_rows*img_cols)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
simple_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None)

model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                   activation='relu',
#                   input_shape=input_shape,kernel_initializer=simple_init))
# model.add(Flatten())

model.add(Reshape((28*28,), input_shape=input_shape))
for i in range(17):
    model.add(Dense(50, activation='relu',kernel_initializer=simple_init))
model.add(Dense(num_classes, activation='softmax',kernel_initializer=simple_init))

if sgd == 'sgd_rg':
    optimizer = SGD_RandomGradient(lr=init_lr)
else:
    optimizer = keras.optimizers.SGD(lr=init_lr)
# optimizer = SGD_RandomGradient(lr=init_lr)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

# plot_model(model, to_file=work_path/'minist.png',show_shapes=True)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=callbacks
          )
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
Test_Acc_Log = '%s\t%.4f\t%.4f'%(exp_name,score[0],score[1])
print(Test_Acc_Log, file=open(work_path.__str__(), 'a'))
