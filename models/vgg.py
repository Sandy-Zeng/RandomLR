import keras
from keras.models import Model
from keras.layers import (Input, Convolution2D, Activation, BatchNormalization,
                          Dropout, MaxPooling2D, ZeroPadding2D, Flatten, Dense)
from keras.regularizers import l2
import keras.layers as layers

#https://github.com/robertomest/convnet-study


def conv_bn_relu(x, num_filters, l2_reg, init='he_normal', border_mode='same',
                 name=None):
    o = layers.Conv2D(num_filters, (3, 3),
                  activation='relu',
                  padding='same',
                  name=name)(x)
    # o = Convolution2D(num_filters, 3, 3, border_mode=border_mode,
    #                   W_regularizer=l2(l2_reg), bias=False,
    #                   init=init, name=name)(x)
    # # o = BatchNormalization(name=name+'_bn')(o)
    # o = Activation('relu', name=name+'_relu')(o)
    return o

def model(input_shape=(32,32,3), l2_reg=5e-4, init='he_normal',num_classes=10):
    # if dataset in ['CIFAR10', 'CIFAR100', 'SVHN']:
    #     x = Input((32, 32, 3))
    # else:
    #     raise ValueError('Model is not defined for dataset: %s' %dataset)
    x = Input(input_shape)

    # Input size is 32x32
    o = conv_bn_relu(x, 64, l2_reg, init=init, name='block1_conv1')
    # o = Dropout(0.3)(o)
    o = conv_bn_relu(o, 64, l2_reg, init=init, name='block1_conv2')
    o = MaxPooling2D()(o)

    # Input size is 16x16
    o = conv_bn_relu(o, 128, l2_reg, init=init, name='block2_conv1')
    # o = Dropout(0.4)(o)
    o = conv_bn_relu(o, 128, l2_reg, init=init, name='block2_conv2')
    o = MaxPooling2D()(o)

    # Input size is 8x8
    o = conv_bn_relu(o, 256, l2_reg, init=init, name='block3_conv1')
    # o = Dropout(0.4)(o)
    o = conv_bn_relu(o, 256, l2_reg, init=init, name='block3_conv2')
    # o = Dropout(0.4)(o)
    o = conv_bn_relu(o, 256, l2_reg, init=init, name='block3_conv3')
    o = MaxPooling2D()(o)

    # Input size is 4x4
    o = conv_bn_relu(o, 512, l2_reg, init=init, name='block4_conv1')
    # o = Dropout(0.4)(o)
    o = conv_bn_relu(o, 512, l2_reg, init=init, name='block4_conv2')
    # o = Dropout(0.4)(o)
    o = conv_bn_relu(o, 512, l2_reg, init=init, name='block4_conv3')
    o = MaxPooling2D()(o)

    o = conv_bn_relu(o, 512, l2_reg, init=init, name='block5_conv1')
    # o = Dropout(0.4)(o)
    o = conv_bn_relu(o, 512, l2_reg, init=init, name='block5_conv2')
    # o = Dropout(0.4)(o)
    o = conv_bn_relu(o, 512, l2_reg, init=init, name='block5_conv3')
    o = MaxPooling2D()(o)

    # Input size is 2x2
    # Manually pad the image to 4x4 and use VALID padding to get it back to 2x2
    # o = ZeroPadding2D(padding=(1,1))(o)
    # o = conv_bn_relu(o, 512, l2_reg, init=init, name='block5_conv1',
    #                  border_mode='valid')
    # # o = Dropout(0.4)(o)
    # o = ZeroPadding2D(padding=(1,1))(o)
    # o = conv_bn_relu(o, 512, l2_reg, init=init, name='block5_conv2',
    #                  border_mode='valid')
    # # o = Dropout(0.4)(o)
    # o = ZeroPadding2D(padding=(1,1))(o)
    # o = conv_bn_relu(o, 512, l2_reg, init=init, name='block5_conv3',
    #                  border_mode='valid')
    # o = MaxPooling2D()(o)

    # Input size is 1x1
    o = Flatten()(o)
    # Classifier
    o = Dropout(0.5)(o)
    o = Dense(512)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Dropout(0.5)(o)

    # if dataset in ['cifar10', 'svhn']:
    #     output_size = 10
    # elif dataset == 'cifar100':
    #     output_size = 100
    output_size = num_classes
    o = Dense(output_size)(o)
    o = Activation('softmax')(o)

    # Classifier

    return Model(input=x, output=o)