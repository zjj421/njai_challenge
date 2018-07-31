#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-7-30
# Task: 
# Insights: 

from datetime import datetime

from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, ZeroPadding2D, concatenate, \
    UpSampling2D, Activation
from keras.models import Model
from keras_applications.densenet import dense_block, DenseNet121, backend, layers
from tqdm import tqdm

from func.config import Config
from func.group_norm import GroupNormalization
from func.se import squeeze_excite_block

CONFIG = Config()
GN_AXIS = 3
CHANNEL_AXIS = GN_AXIS


def conv_block(prev, num_filters, kernel=(3, 3), strides=(1, 1), act='relu', prefix=None):
    name = None
    if prefix is not None:
        name = prefix + '_conv'
    conv = Conv2D(num_filters, kernel, padding='same', kernel_initializer='he_normal', strides=strides, name=name)(prev)
    if prefix is not None:
        name = prefix + '_norm'
    conv = GroupNormalization(name=name, axis=GN_AXIS)(conv)
    if prefix is not None:
        name = prefix + '_act'
    conv = Activation(act, name=name)(conv)
    return conv


def transition_block(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.`
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    # x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
    #                               name=name + '_bn')(x)
    x = GroupNormalization(axis=GN_AXIS, groups=4,
                           scale=False,
                           name=name + '_gn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)
    return x


def get_densenet121_unet_sigmoid_gn(input_shape=(CONFIG.img_h, CONFIG.img_w, CONFIG.img_c),
                                    output_channels=1,
                                    weights='imagenet'):
    blocks = [6, 12, 24, 16]
    img_input = Input(input_shape)

    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)

    x = GroupNormalization(axis=GN_AXIS, groups=16,
                           scale=False,
                           name='conv1/gn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    conv1 = x
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)
    x = dense_block(x, blocks[0], name='conv2')
    conv2 = x
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    conv3 = x
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    conv4 = x
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')
    x = GroupNormalization(axis=GN_AXIS, groups=32,
                           scale=False,
                           name='conv5/gn')(x)
    conv5 = x

    # squeeze and excite block
    conv5 = squeeze_excite_block(conv5)

    conv6 = conv_block(UpSampling2D()(conv5), 320)
    conv6 = concatenate([conv6, conv4], axis=-1)
    conv6 = conv_block(conv6, 320)

    conv7 = conv_block(UpSampling2D()(conv6), 256)
    conv7 = concatenate([conv7, conv3], axis=-1)
    conv7 = conv_block(conv7, 256)

    conv8 = conv_block(UpSampling2D()(conv7), 128)
    conv8 = concatenate([conv8, conv2], axis=-1)
    conv8 = conv_block(conv8, 128)

    conv9 = conv_block(UpSampling2D()(conv8), 96)
    conv9 = concatenate([conv9, conv1], axis=-1)
    conv9 = conv_block(conv9, 96)

    conv10 = conv_block(UpSampling2D()(conv9), 64)
    conv10 = conv_block(conv10, 64)
    res = Conv2D(output_channels, (1, 1), activation='sigmoid')(conv10)
    model = Model(img_input, res)

    if weights == 'imagenet':
        densenet = DenseNet121(input_shape=(input_shape[0], input_shape[1], 3), weights=weights, include_top=False)
        print("Loading imagenet weights.")
        for i in tqdm(range(2, len(densenet.layers) - 1)):
            model.layers[i].set_weights(densenet.layers[i].get_weights())
            model.layers[i].trainable = False

    return model


def get_densenet121_unet_softmax(input_shape, weights='imagenet'):
    blocks = [6, 12, 24, 16]
    img_input = Input(input_shape + (4,))

    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=GN_AXIS, epsilon=1.001e-5,
                           name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    conv1 = x
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)
    x = dense_block(x, blocks[0], name='conv2')
    conv2 = x
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    conv3 = x
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    conv4 = x
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')
    x = BatchNormalization(axis=GN_AXIS, epsilon=1.001e-5,
                           name='bn')(x)
    conv5 = x

    conv6 = conv_block(UpSampling2D()(conv5), 320)
    conv6 = concatenate([conv6, conv4], axis=-1)
    conv6 = conv_block(conv6, 320)

    conv7 = conv_block(UpSampling2D()(conv6), 256)
    conv7 = concatenate([conv7, conv3], axis=-1)
    conv7 = conv_block(conv7, 256)

    conv8 = conv_block(UpSampling2D()(conv7), 128)
    conv8 = concatenate([conv8, conv2], axis=-1)
    conv8 = conv_block(conv8, 128)

    conv9 = conv_block(UpSampling2D()(conv8), 96)
    conv9 = concatenate([conv9, conv1], axis=-1)
    conv9 = conv_block(conv9, 96)

    conv10 = conv_block(UpSampling2D()(conv9), 64)
    conv10 = conv_block(conv10, 64)
    res = Conv2D(3, (1, 1), activation='softmax')(conv10)
    model = Model(img_input, res)

    if weights == 'imagenet':
        densenet = DenseNet121(input_shape=input_shape + (3,), weights=weights, include_top=False)
        w0 = densenet.layers[2].get_weights()
        w = model.layers[2].get_weights()
        w[0][:, :, [0, 1, 2], :] = 0.9 * w0[0][:, :, :3, :]
        w[0][:, :, 3, :] = 0.1 * w0[0][:, :, 1, :]
        model.layers[2].set_weights(w)
        for i in range(3, len(densenet.layers)):
            model.layers[i].set_weights(densenet.layers[i].get_weights())
            model.layers[i].trainable = False

    return model


def __main():
    pass


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
