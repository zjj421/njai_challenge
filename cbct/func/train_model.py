#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-20
# Task: 
# Insights: 

import os
import sys
from datetime import datetime

import keras
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import DepthwiseConv2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import CustomObjectScope, multi_gpu_model
from keras_applications.mobilenet import relu6

from func.model import unet_kaggle, unet
from func.model_inception_resnet_v2 import get_inception_resnet_v2_unet_sigmoid, dice_coef_rounded_ch0, \
    dice_coef_rounded_ch1, sigmoid_dice_loss
from func.utils import mean_iou, prepare_all_data, mean_iou_ch0, mean_iou_ch1
from zf_unet_576_model import dice_coef, dice_coef_loss


# class MultiGpuCbk(keras.callbacks.Callback):
#     def __init__(self, model):
#         self.model_to_save = model
#
#     def on_epoch_end(self, epoch, logs=None):
#         try:
#             self.model_to_save.save('tmp/model_at_epoch_{}.h5'.format(epoch + 1))
#         except:
#             os.makedirs("tmp")
#             self.model_to_save.save('tmp/model_at_epoch_{}.h5'.format(epoch + 1))


class ModelCheckpointMGPU(ModelCheckpoint):
    def __init__(self, original_model, filepath, monitor='val_loss', verbose=0, save_best_only=False,
                 save_weights_only=False, mode='auto', period=1):
        self.original_model = original_model
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def on_epoch_end(self, epoch, logs=None):
        self.model = self.original_model
        super().on_epoch_end(epoch, logs)


def train_generator(model_def, model_saved_path, h5_data_path, batch_size, epochs, model_weights, gpus=1, verbose=1):
    opt = Adam(lr=1e-4, amsgrad=True)
    fit_metrics = [dice_coef_rounded_ch0, dice_coef_rounded_ch1, metrics.binary_crossentropy, mean_iou_ch0,
                   mean_iou_ch1, mean_iou]
    # es = EarlyStopping('val_acc', patience=30, mode="auto", min_delta=0.0)
    # reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=20, verbose=2, epsilon=1e-4,
    #                               mode='auto')

    if model_weights:
        with CustomObjectScope({'relu6': relu6,
                                'DepthwiseConv2D': DepthwiseConv2D}):
            try:
                model = load_model(model_weights)
            except:
                model = model_def
                model.load_weights(model_weights)
            print("Model weights {} have been loaded.".format(model_weights))
    else:
        model = model_def
        print("Model created.")

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
        parallel_model.compile(loss=sigmoid_dice_loss,
                               optimizer=opt,
                               metrics=fit_metrics)
        model.summary()
        cbk1 = ModelCheckpointMGPU(model, model_saved_path, save_best_only=True, monitor='val_loss', mode='min')
        hist = parallel_model.fit_generator(
            train_data_generator,
            validation_data=val_data_generator,
            # steps_per_epoch=count_train // batch_size,
            # validation_steps=count_val // batch_size,
            epochs=epochs,
            callbacks=[cbk1],
            verbose=verbose,
            workers=4,
            use_multiprocessing=True,
            shuffle=True
        )
    else:
        model.compile(loss=sigmoid_dice_loss,
                      optimizer=opt,
                      metrics=fit_metrics)
        model.summary()
        cbk1 = ModelCheckpoint(model_saved_path, save_best_only=True, monitor='val_loss', mode='min')

        hist = model.fit_generator(
            train_data_generator,
            validation_data=val_data_generator,
            steps_per_epoch=None,
            validation_steps=None,
            epochs=epochs,
            callbacks=[cbk1],
            verbose=verbose,
            workers=4,
            use_multiprocessing=True,
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
    model_saved_path = "/home/jzhang/helloworld/mtcnn/cb/model_weights/model_2channel.h5"
    model_trained = "model_weights/model.h5"
    h5_data_path = "/home/jzhang/helloworld/mtcnn/cb/inputs/data.hdf5"
    model_def = unet_kaggle()
    train_generator(model_def, model_saved_path, h5_data_path, batch_size=8, epochs=800, model_weights=model_saved_path,
                    gpus=2,
                    verbose=2)


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    KTF.set_session(sess)

    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
