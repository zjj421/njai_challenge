#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-20
# Task: 
# Insights: 

import os
from datetime import datetime

import gc
import keras.backend as K
import pandas as pd
from keras import metrics
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from func.config import Config
from func.data_io import DataSet
from func.model_densenet import get_densenet121_unet_sigmoid_gn
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
                    csv_log_suffix="0", fold_k="0", random_k_fold=False,
                    input_channels=1, output_channels=2, random_crop_size=(256, 256), mask_nb=0, seed=0

                    ):
    model_weights_root = os.path.dirname(model_saved_path)
    if not os.path.isdir(model_weights_root):
        os.makedirs(model_weights_root)

    learning_rate_scheduler = LearningRateScheduler(schedule=get_learning_rate_scheduler, verbose=0)
    opt = Adam(amsgrad=True)
    # opt = SGD()
    log_path = os.path.join(CONFIG.log_root, "log_" + os.path.splitext(os.path.basename(model_saved_path))[
        0]) + "_" + csv_log_suffix + ".csv"
    if not os.path.isdir(CONFIG.log_root):
        os.makedirs(CONFIG.log_root)

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

    # # prepare train and val data.
    # dataset = DataSet(h5_data_path, val_fold_nb=fold_k, random_k_fold=random_k_fold,
    #                   input_channels=input_channels, output_channels=output_channels,
    #                   random_crop_size=random_crop_size, mask_nb=mask_nb, batch_size=batch_size
    #                   )

    train_ids = ['00000041', '00000042', '00000043', '00000044', '00000045', '00000046', '00000047', '00000048',
                 '00000049', '00000050', '00000051', '00000052', '00000053', '00000054', '00000055', '00000056',
                 '00000057', '00000058', '00000059', '00000060', '00000061', '00000062', '00000063', '00000064',
                 '00000065', '00000066', '00000067', '00000068', '00000069', '00000070', '00000071', '00000072',
                 '00000073', '00000074', '00000075', '00000076', '00000077', '00000078', '00000079', '00000080']

    val_ids = ['00000041', '00000059', '00000074', '00000075']

    dataset = DataSet(h5_data_path, val_fold_nb=fold_k, random_k_fold=random_k_fold,
                      input_channels=input_channels, output_channels=output_channels,
                      random_crop_size=random_crop_size, mask_nb=mask_nb, batch_size=batch_size,
                      train_ids=train_ids, val_ids=val_ids

                      )
    # we create two instances with the same arguments
    # train_data_gen_args = dict(featurewise_center=False,
    #                            featurewise_std_normalization=False,
    #                            rotation_range=15,
    #                            width_shift_range=0.1,
    #                            height_shift_range=0.1,
    #                            horizontal_flip=True,
    #                            fill_mode="nearest",
    #                            shear_range=0.,
    #                            zoom_range=0.15,
    #                            )
    train_data_gen_args = dict(featurewise_center=False,
                               featurewise_std_normalization=False,
                               rotation_range=0,
                               width_shift_range=0.,
                               height_shift_range=0.,
                               horizontal_flip=True,
                               fill_mode="nearest",
                               shear_range=0.,
                               zoom_range=0.,
                               )
    if CONFIG.random_crop_size is None:
        train_data_generator = dataset.get_keras_data_generator(is_train=True,
                                                                keras_data_gen_param=train_data_gen_args,
                                                                seed=seed)
    else:
        train_data_generator = dataset.get_custom_data_generator(is_train=True,
                                                                 keras_data_gen_param=train_data_gen_args,
                                                                 seed=seed)
    # no val augmentation.
    val_data_gen_args = dict(featurewise_center=False,
                             featurewise_std_normalization=False,
                             rotation_range=0.,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             horizontal_flip=False)

    val_data_generator = dataset.get_keras_data_generator(is_train=False, keras_data_gen_param=val_data_gen_args,
                                                          seed=seed)

    model_save_root, model_save_basename = os.path.split(model_saved_path)
    # model_saved_path_best_loss = os.path.join(model_save_root, "best_val_loss_" + model_save_basename)

    if gpus > 1:
        parallel_model = multi_gpu_model(model, gpus=gpus)
        model_checkpoint0 = ModelCheckpointMGPU(model, model_saved_path, save_best_only=True,
                                                save_weights_only=True,
                                                monitor="val_loss",
                                                mode='min')
    else:
        parallel_model = model
        model_checkpoint0 = ModelCheckpoint(model_saved_path, save_best_only=True, save_weights_only=True,
                                            monitor='val_loss', mode='min')
    parallel_model.compile(loss=fit_loss,
                           optimizer=opt,
                           metrics=fit_metrics)
    # model.summary()

    train_steps = dataset.get_train_val_steps(is_train=True)
    val_steps = dataset.get_train_val_steps(is_train=False)
    print("Training ...")
    parallel_model.fit_generator(
        train_data_generator,
        validation_data=val_data_generator,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=[model_checkpoint0, csv_logger, learning_rate_scheduler],
        verbose=verbose,
        workers=1,
        use_multiprocessing=False,
        shuffle=True
    )
    # model_save_root, model_save_basename = os.path.split(model_saved_path)
    # final_model_save_path = os.path.join(model_save_root, "final_" + model_save_basename)
    # model.save_weights(final_model_save_path)

    del model, parallel_model
    K.clear_session()
    gc.collect()


def __main():
    h5_data_path = "/home/jzhang/helloworld/mtcnn/cb/inputs/data_0717.hdf5"

    sub_dir = "20180804_test"

    if not os.path.isdir(sub_dir):
        os.makedirs(sub_dir)
    # model_weights = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/densenet_input1_output2_pretrained_weights.h5"
    # fold_k_lst = ["13", "01", "23", "12", "21", "02", "11", "22"] + ["0", "1", "2"] + list(range(10))
    # fold_k_lst = ["0", "1", "2"]
    fold_k_lst = ["13"]
    random_k_fold = False
    seed = 1248



    for fold_k in fold_k_lst:
        if isinstance(fold_k, int):
            fold_k = str(fold_k)
            random_k_fold = True
        print("Starting training fold", fold_k)
        print("Is random k fold:", random_k_fold)
        model_saved_path = "{}/{}/{}_fold{}_random_kfold_{}_{}i_{}o.h5".format(
            CONFIG.model_weights_save_root,
            sub_dir,
            CONFIG.model_name,
            fold_k,
            random_k_fold * 1,
            CONFIG.input_channels,
            CONFIG.output_channels
        )
        print("Model weights will be saved in '{}'".format(model_saved_path))
        model_def = get_densenet121_unet_sigmoid_gn(input_shape=(None, None, CONFIG.input_channels),
                                                               weights=None,
                                                               output_channels=CONFIG.output_channels)
        # model_def = get_se_inception_resnet_v2_unet_sigmoid_gn(input_shape=(None, None, CONFIG.input_channels),
        #                                                        weights=None,
        #                                                        output_channels=CONFIG.output_channels)

        train_generator(model_def, model_saved_path, h5_data_path, batch_size=2, epochs=120,
                        model_weights=None,
                        gpus=1, verbose=2, csv_log_suffix="1", fold_k=fold_k, random_k_fold=random_k_fold,
                        input_channels=CONFIG.input_channels, output_channels=CONFIG.output_channels,
                        random_crop_size=CONFIG.random_crop_size, mask_nb=CONFIG.mask_nb, seed=seed)
        random_k_fold = False
        seed = seed + 1


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
