from __future__ import print_function

import os
import sys
import numpy as np
import datetime
import dateutil.tz
import argparse
import h5py

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image
import time
from keras.callbacks import LearningRateScheduler
from pathlib import *


parser = argparse.ArgumentParser()
parser.add_argument("--train", help="True for training a new model [False]", action='store_true')
parser.add_argument("--predict", help="True for predicting with an existing model [False]", action='store_true')
parser.add_argument("--epochs", help="Epochs to train [50]", type=int, default=100)
parser.add_argument("--learning_rate", help="Learning rate for the optimizer [0.001]", type=float, default=1e-1)
parser.add_argument("--batch_size", help="The size of batch images [64]", type=int, default=128)
parser.add_argument("--optimizer", help="Optimizer to use. Can be one of: SGD, RMSprop, Adadelta, Adam [Adam]",
                    type=str, default="SGD", choices=set(("SGD", "RMSprop", "Adadelta", "Adam")))
parser.add_argument("--val_size", help="The size of the validation set [5000]", type=int, default=10000)
parser.add_argument("--log_dir", help="Directory name to save the checkpoints and logs [log_dir]",
                    type=str, default="log_dir")
parser.add_argument("--data_set_path", help="Path where data set for training is stored. [svhn_data]",
                    type=str, default="svhn_data")
parser.add_argument("--model", help="Path to model used for prediction. [weights.hdf5]", type=str, default="weights.hdf5")
parser.add_argument("--img_path", help="Path to images to predict. []", type=str,)
parser.add_argument("--distribution_method", type=str,)
parser.add_argument("--random_range", type=int,)
FLAGS = parser.parse_args()


def create_log_dir():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    log_dir = FLAGS.log_dir + "/" + str(sys.argv[0][:-3]) + "_" + timestamp
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # save command line arguments
    # with open(log_dir + "/hyperparameters_" + timestamp + ".csv", "wb") as f:
    #     for arg in FLAGS.__dict__:
    #         print(arg)
    #         f.write(arg + "," + str(FLAGS.__dict__[arg]) + "\n")

    return log_dir

# load svhn data from the specified folder
def load_svhn_data(path, val_size):
    with h5py.File(path+'/SVHN_train.hdf5', 'r') as f:
        shape = f["X"].shape
        x_train = f["X"][:shape[0]-val_size]
        y_train = f["Y"][:shape[0]-val_size].flatten()
        x_val = f["X"][shape[0]-val_size:]
        y_val = f["Y"][shape[0] - val_size:].flatten()

    with h5py.File(path+'/SVHN_test.hdf5', 'r') as f:
        x_test = f["X"][:]
        y_test = f["Y"][:].flatten()

    y_train = keras.utils.to_categorical(y_train, 10)
    y_val = keras.utils.to_categorical(y_val, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def U(tmp_lr,random_range):
    np.random.seed(int(time.time()))
    tmp_lr = np.random.random() * tmp_lr * random_range
    return tmp_lr


def lr_schedule(epoch):
    lr = FLAGS.learning_rate
    epochs = FLAGS.epochs
    # if epoch > epochs * 0.9:
    #     lr *= 0.5e-3
    # elif epoch > epochs * 0.8:
    #     lr *= 1e-3
    # elif epoch > epochs * 0.6:
    #     lr *= 1e-2
    # elif epoch > epochs * 0.4:
    #     lr *= 1e-1
    if epoch > epochs*0.75:
        lr = lr * 1e-2
    elif epoch > epochs*0.5:
        lr = lr * 1e-1
    if FLAGS.distribution_method == 'U' and epoch>epochs*0.5:
        lr = U(lr,FLAGS.random_range)
    print('Learning Rate:',lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)
callbacks = [lr_scheduler]


# build the classification model
def build_model(optimizer, learning_rate, input_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, input_shape=input_shape, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(10, activation='softmax'))

    lr = learning_rate
    optimizers = {"SGD": keras.optimizers.SGD(lr=lr,momentum=0.9), "RMSprop": keras.optimizers.RMSprop(lr=lr),
                  "Adadelta": keras.optimizers.Adadelta(lr=lr), "Adam": keras.optimizers.Adam(lr=lr)}

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizers[optimizer],
                  metrics=['accuracy'])

    return model


# training the model
def train_model(log_dir):
    train_data, val_data, test_data = load_svhn_data(path=FLAGS.data_set_path, val_size=FLAGS.val_size)

    model = build_model(optimizer=FLAGS.optimizer, learning_rate=FLAGS.learning_rate)

    # callback for the training process
    save_model = keras.callbacks.ModelCheckpoint(log_dir+"/weights.hdf5", monitor='val_acc', mode='max', verbose=0,
                                                 save_best_only=True, save_weights_only=False, period=1)
    # early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=0, mode='max')
    # tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=10, batch_size=32, write_graph=True,
    #                                           write_grads=False, write_images=False, embeddings_freq=0,
    #                                           embeddings_layer_names=None, embeddings_metadata=None)

    # train model
    model.fit(train_data[0], train_data[1],
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.epochs,
              verbose=1,
              validation_data=val_data,
              callbacks=[lr_scheduler,save_model])

    # calculate and store test set performance on the model with best validation error
    print("Calculating performance on test set...")
    # model = keras.models.load_model(log_dir+"/weights.hdf5")
    score = model.evaluate(test_data[0], test_data[1], verbose=0)
    print('Test loss: {:.4f}'.format(score[0]))
    print('Test accuracy: {:.4f}'.format(score[1]))
    exp_name = 'SVHN_%s_%d_%d_%.4f_%.4f'%(FLAGS.distribution_method,FLAGS.random_range,FLAGS.epochs,score[0],score[1])
    work_path = Path('/home/ouyangzhihao/Backup/Exp/ZYY/RandomLR/Final/DataSets/res.txt')
    print(exp_name,file=open(work_path.__str__(), 'a'))

    # with open(log_dir+"/test_acc-{:.4f}_test_loss-{:.4f}.txt".format(score[1], score[0]), "wb") as file:
    #     file.write('Test accuracy: {:.4f}\n'.format(score[1]))
    #     file.write('Test loss: {:.4f}'.format(score[0]))


# predict image classes
def predict(model, img_path, batch_size):
    model = keras.models.load_model(model)
    # normalize image pixel values into range [0,1]
    img_generator = image.ImageDataGenerator(preprocessing_function=lambda img: img/255.0)
    validation_generator = img_generator.flow_from_directory(directory=img_path, target_size=(32,32), shuffle=False,
                                                             batch_size=batch_size, color_mode="rgb")

    score = model.evaluate_generator(validation_generator)
    print("Accuracy: {:.4f}".format(score[1]))

is_train = True
if is_train:
    data_set_path = '/home/ouyangzhihao/Backup/dataset/SVHN'
    assert os.path.exists(data_set_path + '/SVHN_train.hdf5'), "There exists no file \"SVHN_train.hdf5\" in {}".\
        format(FLAGS.data_set_path)
    assert os.path.exists(data_set_path + '/SVHN_test.hdf5'), "There exists no file \"SVHN_test.hdf5\" in {}". \
        format(FLAGS.data_set_path)
    log_dir = create_log_dir()
    train_model(log_dir)
elif FLAGS.predict:
    assert FLAGS.img_path is not None, "Please specify the directory in which the images are stored via \"--img_path\"."
    assert os.path.exists(FLAGS.img_path), "The specified path to the images does not exit: {}". \
        format(FLAGS.img_path)
    predict(FLAGS.model, FLAGS.img_path, FLAGS.batch_size)
else:
    print("No valid option chosen. Choose either \"--train\" or \"--predict\".")
    print("Use \"--help\" for an overview of the command line arguments.")