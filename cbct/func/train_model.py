#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-20
# Task: 
# Insights: 

import os
from datetime import datetime

import keras.backend.tensorflow_backend as KTF
import pandas as pd
import tensorflow as tf
from keras import metrics
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.engine.saving import load_model, save_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from tqdm import tqdm

from func.config import Config
from func.data_io import DataSet
from func.model_inception_resnet_v2 import sigmoid_dice_stage1_loss, get_inception_resnet_v2_unet_sigmoid
from zf_unet_576_model import dice_coef

CONFIG = Config()


class ModelCheckpointMGPU(ModelCheckpoint):
    def __init__(self, original_model, filepath, monitor='val_loss', verbose=0, save_best_only=False,
                 save_weights_only=False, mode='auto', period=1):
        self.original_model = original_model
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def on_epoch_end(self, epoch, logs=None):
        self.model = self.original_model
        super().on_epoch_end(epoch, logs)


def get_learning_rate_scheduler(epoch, current_lr):
    df = pd.read_csv(CONFIG.learning_rate_scheduler_csv)
    df.set_index("epoch", inplace=True)
    if epoch in df.index:
        return float(df.loc[epoch])
    else:
        return current_lr


def train_generator(model_def, model_saved_path, h5_data_path, batch_size, epochs, model_weights, gpus=1, verbose=1,
                    csv_log_suffix="0"):
    learning_rate_scheduler = LearningRateScheduler(schedule=get_learning_rate_scheduler, verbose=0)
    opt = Adam(lr=5e-5, amsgrad=True)
    fit_metrics = [dice_coef, metrics.binary_crossentropy, "acc"]
    fit_loss = sigmoid_dice_stage1_loss
    log_path = os.path.join(CONFIG.log_root, "log_" + os.path.splitext(os.path.basename(model_saved_path))[
        0]) + "_" + csv_log_suffix + ".csv"
    csv_logger = CSVLogger(log_path, append=True)

    # fit_metrics = [dice_coef_rounded_ch0, dice_coef_rounded_ch1, metrics.binary_crossentropy, mean_iou_ch0,
    #                mean_iou_ch1, "acc"]
    # fit_metrics = [dice_coef_rounded_ch0, metrics.binary_crossentropy, mean_iou_ch0]
    # es = EarlyStopping('val_acc', patience=30, mode="auto", min_delta=0.0)
    # reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=20, verbose=2, epsilon=1e-4,
    #                               mode='auto')

    if model_weights:
        model = model_def
        print("Loading weights ...")
        try:
            model.load_weights(model_weights, by_name=True)
        except:
            stage1_model = get_inception_resnet_v2_unet_sigmoid(weights=None)
            for i in tqdm(range(2, len(stage1_model.layers) - 1)):
                model.layers[i].set_weights(stage1_model.layers[i].get_weights())
                model.layers[i].trainable = False
            save_model(model, model_saved_path, include_optimizer=False)
        print("Model weights {} have been loaded.".format(model_weights))
        # stage1_model = get_inception_resnet_v2_unet_sigmoid(weights=None)
        # for i in tqdm(range(2, len(stage1_model.layers) - 1)):
        #     model.layers[i].trainable = False
    else:
        model = model_def
        print("Model created.")

    dataset = DataSet(h5_data_path, val_fold_nb=0)
    x_train, y_train = dataset.prepare_2channels_output_data(mode="train")
    print(x_train.shape)
    print(y_train.shape)
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

    x_val, y_val = dataset.prepare_2channels_output_data(mode="val")
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
        parallel_model.compile(loss=fit_loss,
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
            callbacks=[cbk1, csv_logger, learning_rate_scheduler],
            verbose=verbose,
            workers=2,
            use_multiprocessing=True,
            shuffle=True
        )
    else:
        model.compile(loss=fit_loss,
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
            callbacks=[cbk1, csv_logger, learning_rate_scheduler],
            verbose=verbose,
            workers=2,
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
    model_weights = "/home/jzhang/helloworld/mtcnn/cb/model_weights/inception_resnet_v2_all_train_1.h5"
    model_saved_path = "/home/jzhang/helloworld/mtcnn/cb/model_weights/inception_v4_fold0_2channels.h5"
    h5_data_path = "/home/jzhang/helloworld/mtcnn/cb/inputs/data_0717.hdf5"
    model_def = get_inception_resnet_v2_unet_sigmoid(input_shape=(576, 576, 1), weights=None, output_channels=2)
    # model_def = get_inception_resnet_v2_unet_sigmoid_stage1(weights="imagenet")
    train_generator(model_def, model_saved_path, h5_data_path, batch_size=3, epochs=1500, model_weights=model_weights,
                    gpus=1, verbose=2, csv_log_suffix="_")


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
