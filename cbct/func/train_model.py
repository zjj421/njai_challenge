#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-20
# Task: 
# Insights: 

import os
from datetime import datetime

import keras
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import DepthwiseConv2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import CustomObjectScope, multi_gpu_model
from keras_applications.mobilenet import relu6

from func.model import unet_kaggle
from func.utils import mean_iou, prepare_all_data
from zf_unet_576_model import dice_coef, dice_coef_loss


class Multi_Gpu_Cbk(keras.callbacks.Callback):
    def __init__(self, model):
        self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        try:
            self.model_to_save.save('tmp/model_at_epoch_{}.h5'.format(epoch + 1))
        except:
            os.makedirs("tmp")
            self.model_to_save.save('tmp/model_at_epoch_{}.h5'.format(epoch + 1))


def train_generator(model_saved_path, h5_data_path, batch_size, epochs, model_trained, gpus=1):
    opt = Adam(lr=1e-3)
    # es = EarlyStopping('val_acc', patience=30, mode="auto", min_delta=0.0)
    # reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=20, verbose=2, epsilon=1e-4,
    #                               mode='auto')

    if model_trained:
        with CustomObjectScope({'relu6': relu6,
                                'DepthwiseConv2D': DepthwiseConv2D}):
            try:
                model = load_model(model_trained)
            except:
                model = unet_kaggle()
                model.load_weights(model_trained)
    else:
        model = unet_kaggle()

    x_train, y_train = prepare_all_data(h5_data_path, mode="train")
    datagen_train = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        # vertical_flip=True,
        # brightness_range=0.2,
        shear_range=0.05,
        zoom_range=0.05,
    )
    train_data_generator = datagen_train.flow(x_train, y_train, batch_size, shuffle=True)

    x_val, y_val = prepare_all_data(h5_data_path, mode="val")
    datagen_val = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=0.,
        width_shift_range=0.,
        height_shift_range=0.,
        horizontal_flip=False)
    val_data_generator = datagen_val.flow(x_val, y_val, batch_size, shuffle=False)

    if gpus > 1:
        parallel_model = multi_gpu_model(model, gpus=gpus)
        parallel_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["acc"])
        model.summary()
        cbk1 = Multi_Gpu_Cbk(model)
        hist = parallel_model.fit_generator(
            train_data_generator,
            validation_data=val_data_generator,
            # steps_per_epoch=count_train // batch_size,
            # validation_steps=count_val // batch_size,
            epochs=epochs,
            callbacks=[cbk1],
            verbose=1,
            workers=1,
            use_multiprocessing=False,
            shuffle=True
        )
    else:
        # model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["acc", mean_iou])
        model.compile(optimizer=opt, loss=dice_coef_loss, metrics=["acc", dice_coef, mean_iou])
        # model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["acc"])
        model.summary()
        cbk1 = ModelCheckpoint(model_saved_path, save_best_only=True, monitor='val_dice_coef', mode='max')

        hist = model.fit_generator(
            train_data_generator,
            validation_data=val_data_generator,
            steps_per_epoch=None,
            validation_steps=None,
            epochs=epochs,
            callbacks=[cbk1],
            verbose=1,
            workers=1,
            use_multiprocessing=False,
            shuffle=True
        )

    # df = pd.DataFrame.from_dict(hist.history)
    # df.to_csv('tmp/hist.csv', encoding='utf-8', index=False)
    #
    # best_epoch = np.argmax(hist.history.get("val_acc")) + 1
    # print("最好的epoch:", best_epoch)
    # try:
    #     os.rename('tmp/model_at_epoch_{}.h5'.format(best_epoch), model_saved_path)
    # except:
    #     print("rename失败。")


def __main():
    model_saved_path = "model.h5"
    h5_data_path = "/home/jzhang/helloworld/mtcnn/cb/inputs/data.hdf5"
    # model_trained = "/home/jzhang/helloworld/mtcnn/cb/inputs/unet_576.h5"
    train_generator(model_saved_path, h5_data_path, batch_size=8, epochs=60, model_trained=model_saved_path, gpus=1)


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    KTF.set_session(sess)

    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
