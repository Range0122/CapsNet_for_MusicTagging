"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this.

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...

Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card

Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import numpy as np
from keras import models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import Conv2D, Input, TimeDistributed, BatchNormalization, Flatten, GRU, Dense, Add, Activation, \
    Reshape, ReLU, LSTM, Bidirectional, MaxPool2D, Dropout
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capslayers import CapsuleLayer, PrimaryCap, Length, Mask
import config as C
import tensorflow as tf

K.set_image_data_format('channels_last')


def PureCapsNet(input_shape, n_class, routings):
    # K.set_image_dim_ordering('th')

    x = Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = Conv2D(filters=128, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv1')(x)
    conv2 = Conv2D(filters=128, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv2')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv3')(conv2)

    # conv1 = Conv2D(filters=128, kernel_size=9, strides=3, padding='valid', activation='relu', name='conv1')(x)
    # conv2 = Conv2D(filters=128, kernel_size=9, strides=2, padding='valid', activation='relu', name='conv2')(conv1)
    # conv3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv3')(conv2)
    # conv4 = Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv4')(conv3)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv3, dim_capsule=C.DIM_CAPSULE, n_channels=16, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=C.DIM_CAPSULE, routings=routings, name='digitcaps')(
        primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Models for training and evaluation (prediction)
    train_model = models.Model(x, out_caps, name='PureCapsNet')

    return train_model


def MixCapsNet(input_shape, n_class, routings):
    # K.set_image_dim_ordering('th')

    x = Input(shape=input_shape)

    # part1
    conv1 = Conv2D(filters=128, kernel_size=3, strides=2, padding='valid', name='conv1')(x)
    bn1 = BatchNormalization(name='bn1')(conv1)
    relu1 = Activation('relu', name='relu1')(bn1)

    conv2 = Conv2D(filters=128, kernel_size=3, strides=2, padding='valid', name='conv2')(relu1)
    bn2 = BatchNormalization(name='bn2')(conv2)
    relu2 = Activation('relu', name='relu2')(bn2)

    conv3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='valid', name='conv3')(relu2)
    bn3 = BatchNormalization(name='bn3')(conv3)
    relu3 = Activation('relu', name='relu3')(bn3)

    # part2-branch-a
    primarycaps = PrimaryCap(relu3, dim_capsule=C.DIM_CAPSULE, n_channels=16, kernel_size=9, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=C.DIM_CAPSULE, routings=routings, name='digitcaps')(
        primarycaps)
    out_caps = Length(name='capsnet')(digitcaps)

    # part2-branch-b
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', name='conv4')(relu3)
    bn4 = BatchNormalization(name='bn4')(conv4)
    relu4 = Activation('relu', name='relu4')(bn4)

    timedis = TimeDistributed(Flatten(), name='timedis')(relu4)
    gru1 = GRU(32, return_sequences=True, name='gru1')(timedis)
    gru2 = GRU(32, return_sequences=False, name='gru2')(gru1)

    fc1 = Dense(n_class, activation='sigmoid', name='fc1')(gru2)

    add = Add(name='add')([out_caps, fc1])

    # output = Activation('sigmoid', name='output')(add)
    # x = Activation('relu', name='relu')(x)

    train_model = models.Model(x, add, name='MixCapsNet')

    return train_model


def NewMixCapsNet(input_shape, n_class, routings):
    # K.set_image_dim_ordering('th')

    x = Input(shape=input_shape)

    # part1
    conv1 = Conv2D(filters=128, kernel_size=5, strides=2, padding='valid', name='conv1')(x)
    bn1 = BatchNormalization(name='bn1')(conv1)
    relu1 = Activation('relu', name='relu1')(bn1)

    conv2 = Conv2D(filters=128, kernel_size=5, strides=2, padding='valid', name='conv2')(relu1)
    bn2 = BatchNormalization(name='bn2')(conv2)
    relu2 = Activation('relu', name='relu2')(bn2)

    conv3 = Conv2D(filters=128, kernel_size=5, strides=2, padding='valid', name='conv3')(relu2)
    bn3 = BatchNormalization(name='bn3')(conv3)
    relu3 = Activation('relu', name='relu3')(bn3)

    # part2-branch-a
    primarycaps = PrimaryCap(relu3, dim_capsule=C.DIM_CAPSULE, n_channels=16, kernel_size=9, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=C.DIM_CAPSULE, routings=routings, name='digitcaps')(
        primarycaps)
    out_caps = Length(name='capsnet')(digitcaps)

    # part2-branch-b
    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', name='conv4')(relu3)
    bn4 = BatchNormalization(name='bn4')(conv4)
    relu4 = Activation('relu', name='relu4')(bn4)
    drop4 = Dropout(0.3, name='drop4')(relu4)

    conv5 = Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', name='conv5')(drop4)
    bn5 = BatchNormalization(name='bn5')(conv5)
    relu5 = Activation('relu', name='relu5')(bn5)
    drop5 = Dropout(0.3, name='drop5')(relu5)

    conv6 = Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', name='conv6')(drop5)
    bn6 = BatchNormalization(name='bn6')(conv6)
    relu6 = Activation('relu', name='relu6')(bn6)
    drop6 = Dropout(0.3, name='drop6')(relu6)

    timedis = TimeDistributed(Flatten(), name='timedis')(drop6)
    gru1 = GRU(32, return_sequences=True, name='gru1')(timedis)
    gru2 = GRU(32, return_sequences=False, name='gru2')(gru1)

    fc1 = Dense(n_class, activation='sigmoid', name='fc1')(gru2)

    add = Add(name='add')([out_caps, fc1])

    # output = Activation('sigmoid', name='output')(add)
    # x = Activation('relu', name='relu')(x)

    train_model = models.Model(x, add, name='NewMixCapsNet')

    return train_model


def CapsExtractNet(input_shape, n_class, routings):
    # K.set_image_dim_ordering('th')

    x = Input(shape=input_shape)

    conv1 = Conv2D(filters=256, kernel_size=9, strides=4, padding='valid', name='conv1')(x)
    bn1 = BatchNormalization(name='bn1')(conv1)
    relu1 = Activation('relu', name='relu1')(bn1)

    primarycaps = PrimaryCap(relu1, dim_capsule=C.DIM_CAPSULE, n_channels=16, kernel_size=9, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=C.DIM_CAPSULE, routings=routings, name='digitcaps')(
        primarycaps)
    reshape = Reshape((int(digitcaps.shape[1]), int(digitcaps.shape[2]), 1))(digitcaps)

    conv2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', name='conv2')(reshape)
    bn2 = BatchNormalization(name='bn2')(conv2)
    relu2 = Activation('relu', name='relu2')(bn2)

    conv3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', name='conv3')(relu2)
    bn3 = BatchNormalization(name='bn3')(conv3)
    relu3 = Activation('relu', name='relu3')(bn3)

    conv4 = Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv4')(relu3)
    bn4 = BatchNormalization(name='bn4')(conv4)
    relu4 = Activation('relu', name='relu4')(bn4)

    timedis = TimeDistributed(Flatten(), name='timedis')(relu4)
    gru1 = GRU(32, return_sequences=True, name='gru1')(timedis)
    gru2 = GRU(32, return_sequences=False, name='gru2')(gru1)

    fc1 = Dense(64, activation='relu', name='fc1')(gru2)
    fc2 = Dense(n_class, activation='sigmoid', name='fc2')(fc1)

    train_model = models.Model(x, fc2, name='CapsExtractNet')

    return train_model


def BasicCNN(input_shape, n_class):
    # K.set_image_dim_ordering('th')

    x = Input(shape=input_shape)

    conv1 = Conv2D(filters=64, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv1')(x)
    conv2 = Conv2D(filters=128, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv2')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv3')(conv2)

    # fc1 = Dense(64, activation='felu', name='fc1')(conv3)
    fc2 = Dense(n_class, activation='sigmoid', name='output')(conv3)

    train_model = models.Model(x, fc2, name='BasicCNN')

    return train_model


def SmallCapsNet(input_shape, n_class, routings):
    # K.set_image_dim_ordering('th')

    x = Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    # conv1 = Conv2D(filters=128, kernel_size=5, strides=2, padding='valid', activation='relu', name='conv1')(x)
    # conv2 = Conv2D(filters=128, kernel_size=5, strides=2, padding='valid', activation='relu', name='conv2')(conv1)
    # conv3 = Conv2D(filters=256, kernel_size=5, strides=2, padding='valid', activation='relu', name='conv3')(conv2)

    conv1 = Conv2D(filters=64, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv1')(x)
    conv2 = Conv2D(filters=128, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv2')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv3')(conv2)
    # conv4 = Conv2D(filters=128, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv4')(conv3)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv3, dim_capsule=1, n_channels=128, kernel_size=5, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=1, routings=routings, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Models for training and evaluation (prediction)
    train_model = models.Model(x, out_caps, name='SmallCapsNet')

    return train_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    m_positive = 0.9
    # m_positive = 0.85
    m_negative = 1 - m_positive
    L = y_true * K.square(K.maximum(0., m_positive - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - m_negative))

    return K.mean(K.sum(L, 1))


if __name__ == "__main__":
    input_shape = (96, 96, 1)
    n_class = 50
    routings = 3

    # model = PureCapsNet(input_shape, n_class, routings)
    # model = MixCapsNet(input_shape, n_class, routings)
    # model = CapsExtractNet(input_shape, n_class, routings)
    model = NewMixCapsNet(input_shape, n_class, routings)

    model.summary()
