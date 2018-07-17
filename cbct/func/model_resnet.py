#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-7-15
# Task: 
# Insights: 

from datetime import datetime

from func.utils import IMG_H, IMG_W, IMG_C
from keras import Input, Model, layers
from keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, concatenate
from keras.utils import plot_model
from keras_applications import get_keras_submodule
from keras_applications.resnet50 import conv_block, ResNet50, identity_block
from tqdm import tqdm

BN_AXIS = 3
CHANNEL_AXIS = BN_AXIS


def conv_block_custom(prev, num_filters, kernel=(3, 3), strides=(1, 1), act='relu', prefix=None):
    name = None
    if prefix is not None:
        name = prefix + '_conv'
    conv = Conv2D(num_filters, kernel, padding='same', kernel_initializer='he_normal', strides=strides, name=name)(prev)
    if prefix is not None:
        name = prefix + '_norm'
    conv = BatchNormalization(name=name, axis=BN_AXIS)(conv)
    if prefix is not None:
        name = prefix + '_act'
    conv = Activation(act, name=name)(conv)
    return conv




def resnet50_unet_sigmoid(input_shape=(IMG_H, IMG_W, IMG_C), weights='imagenet'):
    inp = Input(input_shape)

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inp)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=BN_AXIS, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    c1 = x
    # print("c1")
    # print(c1.shape)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    c2 = x
    # print("c2")
    # print(c2.shape)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    c3 = x
    # print("c3")
    # print(c3.shape)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    c4 = x
    # print("c4")
    # print(c4.shape)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    c5 = x
    # print("c5")
    # print(c5.shape)

    u6 = conv_block_custom(UpSampling2D()(c5), 1024)
    # print("u6")
    # print(u6.shape)
    u6 = concatenate([u6, c4], axis=-1)
    u6 = conv_block_custom(u6, 1024)

    u7 = conv_block_custom(UpSampling2D()(u6), 512)
    # print("u7")
    # print(u7.shape)
    u7 = concatenate([u7, c3], axis=-1)
    u7 = conv_block_custom(u7, 512)

    u8 = conv_block_custom(UpSampling2D()(u7), 256)
    # print("u8")
    # print(u8.shape)
    u8 = concatenate([u8, c2], axis=-1)
    u8 = conv_block_custom(u8, 256)

    u9 = conv_block_custom(UpSampling2D()(u8), 64)
    # print("u9")
    # print(u9.shape)
    u9 = concatenate([u9, c1], axis=-1)
    u9 = conv_block_custom(u9, 64)

    u10 = conv_block_custom(UpSampling2D()(u9), 32)
    u10 = conv_block_custom(u10, 32)

    res = Conv2D(2, (1, 1), activation='sigmoid')(u10)

    model = Model(inp, res)

    if weights == "imagenet":
        resnet50 = ResNet50(weights=weights, include_top=False,
                                                input_shape=(input_shape[0], input_shape[1], 3))
        # resnet50.summary()
        print("Loading imagenet weitghts ...")
        for i in tqdm(range(3, len(resnet50.layers) - 2)):
            try:
                model.layers[i].set_weights(resnet50.layers[i].get_weights())
                model.layers[i].trainable = False
            except:
                print(resnet50.layers[i].name)
                exit()
        print("imagenet weights have been loaded.")
        del resnet50

    return model


def __main():
    model = resnet50_unet_sigmoid()
    layers = model.layers
    layers = [layer.name for layer in layers]
    model.summary()
    # del model
    model2 = ResNet50(include_top=False, weights=None)
    layers2 = model2.layers
    layers2 = [layer.name for layer in layers2]
    model2.summary()
    print(layers[:5])
    print(layers2[:5])
    # plot_model(model, to_file="/tmp/resnet_50_unet.png", show_shapes=True,
    #            )


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
