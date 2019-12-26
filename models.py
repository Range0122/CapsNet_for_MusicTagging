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
    Reshape, LSTM, MaxPool2D, Dropout, ReLU, GlobalAveragePooling2D, Permute, multiply, GlobalMaxPooling2D, \
    Concatenate, Lambda
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capslayers import CapsuleLayer, PrimaryCap, Length, Mask
import config as C
import tensorflow as tf

K.set_image_data_format('channels_last')


def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def PureCapsNet(input_shape, n_class, routings):
    # K.set_image_dim_ordering('th')

    x = Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = Conv2D(filters=128, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv1')(x)
    conv2 = Conv2D(filters=128, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv2')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv3')(conv2)

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


def Basic_CNN(input_shape, n_class):
    # K.set_image_dim_ordering('th')

    x = Input(input_shape, name='input')

    # Part1
    conv1 = Conv2D(64, (3, 3), padding='same', name='conv1')(x)
    bn1 = BatchNormalization(name='bn1')(conv1)
    relu1 = ReLU()(bn1)
    pool1 = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool1')(relu1)
    drop1 = Dropout(0.1, name='dropout1')(pool1)

    conv2 = Conv2D(128, (3, 3), padding='same', name='conv2')(drop1)
    bn2 = BatchNormalization(name='bn2')(conv2)
    relu2 = ReLU()(bn2)
    pool2 = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool2')(relu2)
    drop2 = Dropout(0.2, name='dropout2')(pool2)

    conv3 = Conv2D(128, (3, 3), padding='same', name='conv3')(drop2)
    bn3 = BatchNormalization(name='bn3')(conv3)
    relu3 = ReLU()(bn3)
    pool3 = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool3')(relu3)
    drop3 = Dropout(0.2, name='dropout3')(pool3)

    conv4 = Conv2D(128, (3, 3), padding='same', name='conv4')(drop3)
    bn4 = BatchNormalization(name='bn4')(conv4)
    relu4 = ReLU()(bn4)
    pool4 = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool4')(relu4)
    drop4 = Dropout(0.3, name='dropout4')(pool4)

    conv5 = Conv2D(128, (3, 3), padding='same', name='conv5')(drop4)
    bn5 = BatchNormalization(name='bn5')(conv5)
    relu5 = ReLU()(bn5)
    pool5 = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool5')(relu5)
    drop5 = Dropout(0.3, name='dropout5')(pool5)

    conv6 = Conv2D(128, (3, 3), padding='same', name='conv6')(drop5)
    bn6 = BatchNormalization(name='bn6')(conv6)
    relu6 = ReLU()(bn6)
    pool6 = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool6')(relu6)
    drop6 = Dropout(0.3, name='dropout6')(pool6)
    flastten = Flatten()(drop6)

    fc1 = Dense(n_class, activation='sigmoid', name='fc1')(flastten)

    train_model = models.Model(x, fc1, name='MixCapsNet')

    return train_model


def TestMixCapsNet(input_shape, n_class, routings):
    # K.set_image_dim_ordering('th')

    x = Input(input_shape, name='input')

    # Part1
    conv1 = Conv2D(64, (3, 3), padding='same', name='conv1')(x)
    bn1 = BatchNormalization(name='bn1')(conv1)
    relu1 = ReLU()(bn1)
    pool1 = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool1')(relu1)
    drop1 = Dropout(0.3, name='dropout1')(pool1)
    # cbam1 = cbam_block(drop1, 8)

    conv2 = Conv2D(128, (3, 3), padding='same', name='conv2')(drop1)
    # conv2 = Conv2D(128, (3, 3), padding='same', name='conv2')(cbam1)
    bn2 = BatchNormalization(name='bn2')(conv2)
    relu2 = ReLU()(bn2)
    pool2 = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool2')(relu2)
    drop2 = Dropout(0.3, name='dropout2')(pool2)
    # cbam2 = cbam_block(drop2, 8)

    conv3 = Conv2D(128, (3, 3), padding='same', name='conv3')(drop2)
    # conv3 = Conv2D(128, (3, 3), padding='same', name='conv3')(cbam2)
    bn3 = BatchNormalization(name='bn3')(conv3)
    relu3 = ReLU()(bn3)
    pool3 = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool3')(relu3)
    drop3 = Dropout(0.3, name='dropout3')(pool3)
    # cbam3 = cbam_block(drop3, 8)

    conv4 = Conv2D(128, (3, 3), padding='same', name='conv4')(drop3)
    # conv4 = Conv2D(128, (3, 3), padding='same', name='conv4')(cbam3)
    bn4 = BatchNormalization(name='bn4')(conv4)
    relu4 = ReLU()(bn4)
    pool4 = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool4')(relu4)
    drop4 = Dropout(0.3, name='dropout4')(pool4)
    # cbam4 = cbam_block(drop4, 8)

    conv5 = Conv2D(128, (3, 3), padding='same', name='conv5')(drop4)
    # conv5 = Conv2D(128, (3, 3), padding='same', name='conv5')(cbam4)
    bn5 = BatchNormalization(name='bn5')(conv5)
    relu5 = ReLU()(bn5)
    pool5 = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool5')(relu5)
    drop5 = Dropout(0.3, name='dropout5')(pool5)
    # cbam5 = cbam_block(drop5, 8)

    conv6 = Conv2D(128, (3, 3), padding='same', name='conv6')(drop5)
    # conv6 = Conv2D(128, (3, 3), padding='same', name='conv6')(cbam5)
    bn6 = BatchNormalization(name='bn6')(conv6)
    relu6 = ReLU()(bn6)
    pool6 = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool6')(relu6)
    drop6 = Dropout(0.3, name='dropout6')(pool6)
    # cbam6 = cbam_block(drop6, 8)

    conv7 = Conv2D(128, (3, 3), padding='same', name='conv7')(drop6)
    # conv7 = Conv2D(128, (3, 3), padding='same', name='conv7')(cbam6)
    bn7 = BatchNormalization(name='bn7')(conv7)
    relu7 = ReLU()(bn7)
    pool7 = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool7')(relu7)
    drop7 = Dropout(0.3, name='dropout7')(pool7)
    # cbam7 = cbam_block(drop7, 8)

    # Part2-branch-a
    primarycaps = PrimaryCap(drop7, dim_capsule=8, n_channels=16, kernel_size=3, strides=2, padding='same')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=32, routings=routings, name='digitcaps')(primarycaps)
    out_caps = Length(name='capsnet')(digitcaps)
    # fc1 = Dense(n_class, activation='sigmoid', name='fc1')(out_caps)

    # Part2-branch-b
    flastten = Flatten()(drop7)
    fc1 = Dense(n_class, activation='sigmoid', name='fc1')(flastten)

    # flastten = Flatten()(drop7)
    # fc1 = Dense(128, activation='relu', name='fc1')(flastten)
    # fc2 = Dense(n_class, activation='sigmoid', name='fc2')(fc1)

    # timedis = TimeDistributed(Flatten(), name='timedis')(drop7)
    # gru1 = LSTM(64, return_sequences=True, name='gru1')(timedis)
    # gru2 = LSTM(64, return_sequences=False, name='gru2')(gru1)
    # fc1 = Dense(n_class, activation='sigmoid', name='fc1')(gru2)

    # Part3
    add = Add(name='add')([out_caps, fc1])

    train_model = models.Model(x, add, name='TestMixCapsNet')

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
    input_shape = (96, 1366, 1)
    n_class = 50
    routings = 3

    # model = PureCapsNet(input_shape, n_class, routings)
    # model = MixCapsNet(input_shape, n_class, routings)
    # model = CapsExtractNet(input_shape, n_class, routings)
    model = TestModel(input_shape, n_class)

    model.summary()
