#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-20
# Task: 
# Insights: 

import os
from datetime import datetime

import keras.backend as K
import pandas as pd
from keras import metrics
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model

from func.config import Config
from func.data_io import DataSet
from func.model_inception_resnet_v2 import get_inception_resnet_v2_unet_sigmoid, \
    dice_coef_rounded_ch0, dice_coef_rounded_ch1, sigmoid_dice_loss, binary_acc_ch0, dice_coef, \
    sigmoid_dice_loss_1channel_output
from func.model_inception_resnet_v2_gn import get_inception_resnet_v2_unet_sigmoid_gn
from func.model_se_inception_resnet_v2_gn import get_se_inception_resnet_v2_unet_sigmoid_gn
from func.utils import mean_iou_ch0
import numpy as np

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
                    csv_log_suffix="0", fold_k=0):
    learning_rate_scheduler = LearningRateScheduler(schedule=get_learning_rate_scheduler, verbose=0)
    opt = Adam(amsgrad=True)
    # opt = SGD()
    log_path = os.path.join(CONFIG.log_root, "log_" + os.path.splitext(os.path.basename(model_saved_path))[
        0]) + "_" + csv_log_suffix + ".csv"
    if os.path.isfile(log_path):
        print("Log file exists.")
        # exit()
    csv_logger = CSVLogger(log_path, append=False)

    # tensorboard = TensorBoard(log_dir='/home/jzhang/helloworld/mtcnn/cb/logs/tensorboard', write_images=True)

    # fit_metrics = [dice_coef, metrics.binary_crossentropy, binary_acc_ch0]
    # fit_loss = sigmoid_dice_loss_1channel_output

    fit_loss = sigmoid_dice_loss
    fit_metrics = [dice_coef_rounded_ch0, dice_coef_rounded_ch1, metrics.binary_crossentropy, mean_iou_ch0,
                   binary_acc_ch0]
    # fit_metrics = [dice_coef_rounded_ch0, metrics.binary_crossentropy, mean_iou_ch0]
    # es = EarlyStopping('val_acc', patience=30, mode="auto", min_delta=0.0)
    # reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=20, verbose=2, epsilon=1e-4,
    #                               mode='auto')

    if model_weights:
        model = model_def
        print("Loading weights ...")
        model.load_weights(model_weights, by_name=True, skip_mismatch=True)
        print("Model weights {} have been loaded.".format(model_weights))
    else:
        model = model_def
        print("Model created.")

    # prepare train and val data.
    dataset = DataSet(h5_data_path, val_fold_nb=fold_k)
    # x_train, y_train = dataset.prepare_stage2_data(is_train=True)
    x_train, y_train = dataset.prepare_1i_2o_data(is_train=True)
    print(x_train.shape)
    print(y_train.shape)
    # x_val, y_val = dataset.prepare_stage2_data(is_train=False)
    x_val, y_val = dataset.prepare_1i_2o_data(is_train=False)
    print(x_val.shape)
    print(y_val.shape)
    # we create two instances with the same arguments
    # train_data_gen_args = dict(featurewise_center=False,
    #                            featurewise_std_normalization=False,
    #                            rotation_range=15,
    #                            width_shift_range=0.1,
    #                            height_shift_range=0.1,
    #                            horizontal_flip=True,
    #                            # vertical_flip=True,
    #                            # brightness_range=0.1,
    #                            shear_range=0.1,
    #                            zoom_range=0.1, )
    train_data_gen_args = dict(featurewise_center=False,
                               featurewise_std_normalization=False,
                               rotation_range=20,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               horizontal_flip=True,
                               fill_mode="nearest",
                               shear_range=0.,
                               zoom_range=0.15, )
    train_image_datagen = ImageDataGenerator(**train_data_gen_args)
    train_mask_datagen = ImageDataGenerator(**train_data_gen_args)
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 12488421
    # train_image_datagen.fit(x_train, augment=True, seed=seed)
    # train_mask_datagen.fit(y_train, augment=True, seed=seed)
    train_image_generator = train_image_datagen.flow(x_train, None, batch_size, shuffle=True, seed=seed)
    train_mask_generator = train_mask_datagen.flow(y_train, None, batch_size, shuffle=True, seed=seed)
    # combine generators into one which yields image and masks
    train_data_generator = zip(train_image_generator, train_mask_generator)

    # no val augmentation.
    val_data_gen_args = dict(featurewise_center=False,
                             featurewise_std_normalization=False,
                             rotation_range=0.,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             horizontal_flip=False)
    val_image_datagen = ImageDataGenerator(**val_data_gen_args)
    val_mask_datagen = ImageDataGenerator(**val_data_gen_args)
    val_image_generator = val_image_datagen.flow(x_val, None, batch_size, shuffle=False, seed=seed)
    val_mask_generator = val_mask_datagen.flow(y_val, None, batch_size, shuffle=False, seed=seed)
    val_data_generator = zip(val_image_generator, val_mask_generator)

    model_save_root, model_save_basename = os.path.split(model_saved_path)
    model_saved_path0 = os.path.join(model_save_root, "best_val_loss_" + model_save_basename)
    model_saved_path1 = os.path.join(model_save_root, "best_val_acc_" + model_save_basename)

    if gpus > 1:
        parallel_model = multi_gpu_model(model, gpus=gpus)
        model_checkpoint0 = ModelCheckpointMGPU(model, model_saved_path0, save_best_only=True, save_weights_only=True,
                                                monitor="val_loss",
                                                mode='min')
        model_checkpoint1 = ModelCheckpointMGPU(model, model_saved_path1, save_best_only=True, save_weights_only=True,
                                                monitor="val_binary_acc_ch0",
                                                mode='max')
    else:
        parallel_model = model
        model_checkpoint0 = ModelCheckpoint(model_saved_path0, save_best_only=True, save_weights_only=True,
                                            monitor='val_loss', mode='min')
        model_checkpoint1 = ModelCheckpoint(model_saved_path1, save_best_only=True, save_weights_only=True,
                                            monitor='val_binary_acc_ch0', mode='max')
    parallel_model.compile(loss=fit_loss,
                           optimizer=opt,
                           metrics=fit_metrics)
    model.summary()

    count_train = x_train.shape[0]
    count_val = x_val.shape[0]
    parallel_model.fit_generator(
        train_data_generator,
        validation_data=val_data_generator,
        steps_per_epoch=count_train // batch_size,
        validation_steps=count_val // batch_size,
        epochs=epochs,
        callbacks=[model_checkpoint0, model_checkpoint1, csv_logger, learning_rate_scheduler],
        verbose=verbose,
        workers=4,
        use_multiprocessing=True,
        shuffle=True
    )
    # model_save_root, model_save_basename = os.path.split(model_saved_path)
    final_model_save_path = os.path.join(model_save_root, "final_" + model_save_basename)
    parallel_model.save_weights(final_model_save_path)


def __main():
    h5_data_path = "/home/jzhang/helloworld/mtcnn/cb/inputs/data_0717.hdf5"
    os.makedirs("/home/jzhang/helloworld/mtcnn/cb/model_weights/20180727")
    model_weights = "/home/jzhang/helloworld/mtcnn/cb/model_weights/inception_resnet_v2_input1_output2_pretrained_weights.h5"

    model_saved_path = "/home/jzhang/helloworld/mtcnn/cb/model_weights/20180727/se_inception_resnet_v2_gn_fold0_1i_2o_20180727.h5"
    model_def = get_se_inception_resnet_v2_unet_sigmoid_gn(input_shape=(None, None, 1), weights=None,
                                                           output_channels=2)
    train_generator(model_def, model_saved_path, h5_data_path, batch_size=4, epochs=600,
                    model_weights=model_weights,
                    gpus=2, verbose=2, csv_log_suffix="0", fold_k=0)
    K.clear_session()

    model_saved_path = "/home/jzhang/helloworld/mtcnn/cb/model_weights/20180727/se_inception_resnet_v2_gn_fold1_1i_2o_20180727.h5"
    model_def = get_se_inception_resnet_v2_unet_sigmoid_gn(input_shape=(None, None, 1), weights=None,
                                                           output_channels=2)
    train_generator(model_def, model_saved_path, h5_data_path, batch_size=4, epochs=600,
                    model_weights=model_weights,
                    gpus=2, verbose=2, csv_log_suffix="0", fold_k=1)
    K.clear_session()

    model_saved_path = "/home/jzhang/helloworld/mtcnn/cb/model_weights/20180727/se_inception_resnet_v2_gn_fold2_1i_2o_20180727.h5"
    model_def = get_se_inception_resnet_v2_unet_sigmoid_gn(input_shape=(None, None, 1), weights=None,
                                                           output_channels=2)
    train_generator(model_def, model_saved_path, h5_data_path, batch_size=4, epochs=600,
                    model_weights=model_weights,
                    gpus=2, verbose=2, csv_log_suffix="0", fold_k=2)
    K.clear_session()


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2"
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # KTF.set_session(sess)

    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
