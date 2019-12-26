# !/usr/bin/env python
# -*- coding: UTF-8 -*-
# __author__: Qmh
# __file_name__: models.py
# __time__: 2019:06:27:19:51

import keras.backend as K
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPool2D, Dropout, Activation, merge, ZeroPadding2D
from keras.layers import Dense, Lambda, Add, GlobalAveragePooling2D, ZeroPadding2D, Multiply, GlobalMaxPool2D
from keras.regularizers import l2
from keras import Model
import constants as c
from keras.layers.core import Permute
from keras import regularizers
from keras.layers import Conv1D, MaxPool1D, LSTM
from keras import initializers
from keras.layers import GlobalMaxPool1D, Permute
from keras.layers import GRU, TimeDistributed, Flatten, LeakyReLU, ELU


# Resblcok
def res_conv_block(x, filters, strides, name):
    filter1, filer2, filter3 = filters
    # block a
    x = Conv2D(filter1, (1, 1), strides=strides, kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY),
               name=f'{name}_conva')(x)
    x = BatchNormalization(name=f'{name}_bna')(x)
    x = Activation('relu', name=f'{name}_relua')(x)
    # block b
    x = Conv2D(filer2, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY),
               name=f'{name}_convb')(x)
    x = BatchNormalization(name=f'{name}_bnb')(x)
    x = Activation('relu', name=f'{name}_relub')(x)
    # block c
    x = Conv2D(filter3, (1, 1), name=f'{name}_convc', kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY))(x)
    x = BatchNormalization(name=f'{name}_bnc')(x)
    # shortcut
    shortcut = Conv2D(filter3, (1, 1), strides=strides, name=f'{name}_shcut',
                      kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY))(x)
    shortcut = BatchNormalization(name=f'{name}_stbn')(x)
    x = Add(name=f'{name}_add')([x, shortcut])
    x = Activation('relu', name=f'{name}_relu')(x)
    return x


def basic_block(x, filters, strides, name):
    # block a
    x = Conv2D(filters, (3, 3), strides=strides, padding='same',
               kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY), name=f'{name}_conva')(x)
    x = BatchNormalization(name=f'{name}_bna')(x)
    x = Activation('relu', name=f'{name}_relua')(x)
    # block b
    x = Conv2D(filters, (3, 3), strides=(1, 1), padding='same',
               kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY), name=f'{name}_convb')(x)
    x = BatchNormalization(name=f'{name}_bnb')(x)
    # shortcut
    shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same', name=f'{name}_shcut')(x)
    shortcut = BatchNormalization(name=f'{name}_stbn')(x)
    x = Add(name=f'{name}_add')([x, shortcut])
    x = Activation('relu', name=f'{name}_relu')(x)
    return x


# ResNet
def ResNet_34(input_shape):
    x_in = Input(input_shape, name='input')
    x = Permute((2, 1, 3), name='permute')(x_in)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1',
               kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY))(x)
    x = BatchNormalization(name="bn1")(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='pool1')(x)

    for i in range(1, 4):
        x = basic_block(x, 64, (1, 1), name=f'block{i}')

    for i in range(4, 8):
        strides = (2, 2) if i == 4 else (1, 1)
        x = basic_block(x, 128, strides, name=f'block{i}')

    for i in range(8, 14):
        strides = (2, 2) if i == 8 else (1, 1)
        x = basic_block(x, 256, strides, name=f'block{i}')

    for i in range(14, 17):
        strides = (2, 2) if i == 14 else (1, 1)
        x = basic_block(x, 512, strides, name=f'block{i}')

    x = Conv2D(512, (x.shape[1].value, 1), name='fc6')(x)

    # avgpool
    x = Lambda(lambda y: K.mean(y, axis=[1, 2]), name='avgpool')(x)

    x = Dense(512, name='fc1')(x)
    x = BatchNormalization(name='bn_fc1')(x)
    x = Activation('relu', name='relu_fc1')(x)

    model = Model(inputs=[x_in], outputs=[x], name='ResCNN')
    # model.summary()
    return model


# ResNet
def ResNet_50(input_shape):
    x_in = Input(input_shape, name='input')
    x = Permute((2, 1, 3), name='permute')(x_in)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1',
               kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY))(x)
    x = BatchNormalization(name="bn1")(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='pool1')(x)

    x = res_conv_block(x, (64, 64, 256), (1, 1), name='block1')
    x = res_conv_block(x, (64, 64, 256), (1, 1), name='block2')
    x = res_conv_block(x, (64, 64, 256), (1, 1), name='block3')

    x = res_conv_block(x, (128, 128, 512), (1, 1), name='block4')
    x = res_conv_block(x, (128, 128, 512), (1, 1), name='block5')
    x = res_conv_block(x, (128, 128, 512), (1, 1), name='block6')
    x = res_conv_block(x, (128, 128, 512), (2, 2), name='block7')

    x = Conv2D(512, (x.shape[1].value, 1), name='fc6')(x)
    x = BatchNormalization(name="bn_fc6")(x)
    x = Activation('relu', name='relu_fc6')(x)
    # avgpool
    # x = GlobalAveragePooling2D(name='avgPool')(x)
    x = Lambda(lambda y: K.mean(y, axis=[1, 2]), name='avgpool')(x)

    model = Model(inputs=[x_in], outputs=[x], name='ResCNN')
    # model.summary()
    return model


def squeeze_excitation(x, reduction_ratio, name):
    out_dim = int(x.shape[-1].value)
    x = GlobalAveragePooling2D(name=f'{name}_squeeze')(x)
    x = Dense(out_dim // reduction_ratio, activation='relu', name=f'{name}_ex0')(x)
    x = Dense(out_dim, activation='sigmoid', name=f'{name}_ex1')(x)
    return x


def conv_block(x, filters, kernal_size, strides, name, stage, i, padding='same'):
    x = Conv2D(filters, kernal_size, strides=strides, padding=padding, name=f'{name}_conv{stage}_{i}',
               kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY))(x)
    x = BatchNormalization(name=f'{name}_bn{stage}_{i}')(x)
    if stage != 'c':
        x = ELU(name=f'{name}_relu{stage}_{i}')(x)
    return x


def residual_block(x, outdim, strides, name):
    input_dim = int(x.shape[-1].value)
    shortcut = Conv2D(outdim, kernel_size=(1, 1), strides=strides, name=f'{name}_scut_conv',
                      kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY))(x)
    shortcut = BatchNormalization(name=f'{name}_scut_norm')(shortcut)

    for i in range(c.BLOCK_NUM):
        if i > 0:
            strides = 1
            x = Dropout(c.DROPOUT, name=f'{name}_drop{i - 1}')(x)

        x = conv_block(x, outdim // 4, (1, 1), strides, name, 'a', i, padding='valid')
        x = conv_block(x, outdim // 4, (3, 3), (1, 1), name, 'b', i, padding='same')
        x = conv_block(x, outdim, (1, 1), (1, 1), name, 'c', i, padding='valid')
        x = ELU(name=f'{name}_relu{i}')(x)
    # add SE
    x = Multiply(name=f'{name}_scale')([x, squeeze_excitation(x, c.REDUCTION_RATIO, name)])
    x = Add(name=f'{name}_scut')([shortcut, x])
    x = ELU(name=f'{name}_relu')(x)
    # x = Activation('relu',name=f'{name}_relu')(x)
    return x


# proposed model v4.0 timit libri
def SE_ResNet(input_shape):
    # first layer
    x_in = Input(input_shape, name='input')
    #  f,t,c
    # x = Permute((2,1,3),name='permute')(x_in)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1',
               kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY))(x_in)
    x = BatchNormalization(name='bn1')(x)
    x = ELU(name=f'relu1')(x)

    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    x = residual_block(x, outdim=256, strides=(2, 2), name='block2')
    x = residual_block(x, outdim=256, strides=(2, 2), name='block3')
    x = residual_block(x, outdim=512, strides=(2, 2), name='block6')
    x = residual_block(x, outdim=512, strides=(2, 2), name='block7')

    x = residual_block(x, outdim=1024, strides=(2, 2), name='block8')
    x = residual_block(x, outdim=1024, strides=(2, 2), name='block9')

    x = Lambda(lambda y: K.mean(y, axis=[1, 2]), name='average')(x)

    x = Dense(1024, name='fc1')(x)
    x = BatchNormalization(name='bn_fc1')(x)
    x = ELU(name=f'relu_fc1')(x)

    return Model(inputs=[x_in], outputs=[x], name='SEResNet')


# deep speaker
def clipped_relu(inputs):
    return Lambda(lambda y: K.minimum(K.maximum(y, 0), 20))(inputs)


def identity_block(x_in, kernel_size, filters, name):
    x = Conv2D(filters, kernel_size=kernel_size, strides=(1, 1),
               padding='same', kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY),
               name=f'{name}_conva')(x_in)
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = clipped_relu(x)
    x = Conv2D(filters, kernel_size=kernel_size, strides=(1, 1),
               padding='same', kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY),
               name=f'{name}_convb')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)
    x = Add(name=f'{name}_add')([x, x_in])
    x = clipped_relu(x)
    return x


def Deep_speaker_model(input_shape):
    def conv_and_res_block(x_in, filters):
        x = Conv2D(filters, kernel_size=(5, 5), strides=(2, 2),
                   padding='same', kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY),
                   name=f'conv_{filters}-s')(x_in)
        x = BatchNormalization(name=f'conv_{filters}-s_bn')(x)
        x = clipped_relu(x)
        for i in range(3):
            x = identity_block(x, kernel_size=(3, 3), filters=filters, name=f'res{filters}_{i}')
        return x

    x_in = Input(input_shape, name='input')
    # x = Permute((2,1,3),name='permute')(x_in)
    x = conv_and_res_block(x_in, 64)
    # x = conv_and_res_block(x,128)
    x = conv_and_res_block(x, 256)
    x = conv_and_res_block(x, 512)
    # average
    x = Lambda(lambda y: K.mean(y, axis=[1, 2]), name='avgpool')(x)
    # affine
    x = Dense(512, name='affine')(x)
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)
    model = Model(inputs=[x_in], outputs=[x], name='deepspeaker')
    return model


# proposed model
def Baseline_GRU(input_shape):
    # first layer
    x_in = Input(input_shape, name='input')
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1')(x_in)
    x = BatchNormalization(name='bn1')(x)
    x = ELU(name='relu1')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = ELU(name='relu2')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

    x = TimeDistributed(Flatten(), name='timedis1')(x)
    x = GRU(512, return_sequences=True, name='gru1')(x)
    x = GRU(512, return_sequences=True, name='gru2')(x)
    x = GRU(512, return_sequences=False, name='gru4')(x)

    x = Dense(512, name='fc2', activation='relu')(x)
    x = BatchNormalization(name='fc_norm')(x)
    x = ELU(name='relu3')(x)

    return Model(inputs=[x_in], outputs=[x], name='Baseline_GRU')


## new basic model
def bottle_neck(x, width, name, strides=(1, 1), downsample=False):
    expansion = 4
    # input_dim = int(x.shape[-1].value)

    if downsample:
        identity = Conv2D(width * expansion, kernel_size=(1, 1), strides=strides, padding='same',
                          name=f'{name}_identity',
                          kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY))(x)
        identity = BatchNormalization(name=f'{name}_bn0')(identity)
    else:
        identity = x

    # conv 1x1
    x = Conv2D(width, kernel_size=(1, 1), strides=(1, 1), name=f'{name}_conv1',
               kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY))(x)
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = Activation('relu', name=f'{name}_relu1')(x)

    # conv 3x3
    x = Conv2D(width, kernel_size=(3, 3), strides=strides, padding='same', name=f'{name}_conv2',
               kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY))(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)
    x = Activation('relu', name=f'{name}_relu2')(x)

    # conv 1x1
    x = Conv2D(width * expansion, kernel_size=(1, 1), strides=(1, 1), name=f'{name}_conv3',
               kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY))(x)
    x = BatchNormalization(name=f'{name}_bn3')(x)

    x = Add(name=f'{name}_scut')([identity, x])
    x = Activation('relu', name=f'{name}_relu3')(x)
    return x


def Make_layer(x, width, blocks, name, strides=(1, 1)):
    # 块中的第一层 需要downsample 和 strides
    x = bottle_neck(x, width, name=f'{name}_layer0', strides=strides, downsample=True)
    # 后两层
    for i in range(1, blocks):
        x = bottle_neck(x, width, name=f'{name}_layer{i}')
    return x


def Basic_Model(input_shape, layers_num):
    x_in = Input(input_shape, name='input')
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv1',
               kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY))(x_in)
    X = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='relu1')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(x)

    x = Make_layer(x, 64, layers_num[0], name='block0')
    x = Make_layer(x, 128, layers_num[1], name='block1', strides=(2, 2))
    x = Make_layer(x, 256, layers_num[2], name='block2', strides=(2, 2))

    # x = Make_layer(x,512,layers_num[3],name='block3',strides=(2,2))
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(1024, name='fc1')(x)
    x = BatchNormalization(name='bn_fc1')(x)
    x = Activation('relu', name='fc1_relu')(x)
    return Model(x_in, x, name='BasicModule')


# proposed model

def channel_attention(x, reduction_ratio, name):
    out_dim = int(x.shape[-1].value)
    avg_out = GlobalAveragePooling2D(name=f'{name}_avg_squeeze')(x)
    avg_out = Dense(out_dim // reduction_ratio, activation='relu', name=f'{name}_avg_ex')(avg_out)

    max_out = GlobalMaxPool2D(name=f'{name}_max_squeeze')(x)
    max_out = Dense(out_dim // reduction_ratio, activation='relu', name=f'{name}_max_ex')(max_out)

    out = Add(name=f'{name}_add')([max_out, avg_out])
    out = Dense(out_dim, activation='sigmoid', name=f'{name}_scale')(out)

    x = Multiply(name=f'{name}_mul')([out, x])
    return x


def Spatial_attention(x, name, kernel_size=7):
    assert kernel_size in (3, 7), 'kernel_size must be 3 or 7'
    # padding = 3 if kernel_size==7 else 1

    avg_out = Lambda(lambda y: K.mean(y, axis=3, keepdims=True), name=f'{name}_avgpool')(x)  # (B, W, H, 1)
    max_out = Lambda(lambda y: K.max(x, axis=3, keepdims=True), name=f'{name}_maxpool')(x)  # (B, W, H, 1)

    out = merge.Concatenate(name=f'{name}_concat')([avg_out, max_out])  # (B, W, H, 2)

    out = Conv2D(1, kernel_size, padding='same', name=f'{name}_conv', activation='sigmoid')(out)

    x = Multiply(name=f'{name}_mul')([out, x])

    return x


def new_bottle_neck(x, width, name, strides=(1, 1), downsample=False):
    expansion = 4
    # input_dim = int(x.shape[-1].value)

    if downsample:
        identity = Conv2D(width * expansion, kernel_size=(1, 1), strides=strides, padding='same',
                          name=f'{name}_identity',
                          kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY))(x)
        identity = BatchNormalization(name=f'{name}_bn0')(identity)
    else:
        identity = x

    # conv 1x1
    x = Conv2D(width, kernel_size=(1, 1), strides=(1, 1), name=f'{name}_conv1',
               kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY))(x)
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = Activation('relu', name=f'{name}_relu1')(x)

    # conv 3x3
    x = Conv2D(width, kernel_size=(3, 3), strides=strides, padding='same', name=f'{name}_conv2',
               kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY))(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)
    x = Activation('relu', name=f'{name}_relu2')(x)

    # conv 1x1
    x = Conv2D(width * expansion, kernel_size=(1, 1), strides=(1, 1), name=f'{name}_conv3',
               kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY))(x)
    x = BatchNormalization(name=f'{name}_bn3')(x)

    x = Add(name=f'{name}_scut')([identity, x])
    x = Activation('relu', name=f'{name}_relu3')(x)
    return x


def new_Make_layer(x, width, blocks, name, strides=(1, 1)):
    # 块中的第一层 需要downsample 和 strides
    x = new_bottle_neck(x, width, name=f'{name}_layer0', strides=strides, downsample=True)
    # 后两层
    for i in range(1, blocks):
        x = new_bottle_neck(x, width, name=f'{name}_layer{i}', strides=strides, downsample=True)
    return x


def Proposed_Model(input_shape, layers_num):
    x_in = Input(input_shape, name='input')
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv1',
               kernel_regularizer=regularizers.l2(l=c.WEIGHT_DECAY))(x_in)
    X = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='relu1')(x)

    # add attention
    x = channel_attention(x, c.REDUCTION_RATIO, name='first_ca')
    x = Spatial_attention(x, name='first_sa')

    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(x)

    x = new_Make_layer(x, 64, layers_num[0], name='block0', strides=(1, 1))
    x = new_Make_layer(x, 128, layers_num[1], name='block1', strides=(2, 2))
    x = new_Make_layer(x, 256, layers_num[2], name='block2', strides=(2, 2))

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(1024, name='fc1')(x)
    x = BatchNormalization(name='bn_fc1')(x)
    x = Activation('relu', name='fc1_relu')(x)

    x = Dropout(0.5, name='drop3')(x)
    return Model(x_in, x, name='ProposedModle')


if __name__ == "__main__":
    # model = ResNet(c.INPUT_SHPE)
    # model = vggvox1_cnn((299,40,1))
    # model = Deep_speaker_model(c.INPUT_SHPE)
    # model = SE_ResNet(c.INPUT_SHPE)
    # model = RWCNN_LSTM((59049,1))
    # model = Basic_Model(c.INPUT_SHPE,[2,3,2])
    # model = Baseline_GRU(c.INPUT_SHPE)
    model = Proposed_Model(c.INPUT_SHPE, [2, 2, 1])
    print(model.summary())
