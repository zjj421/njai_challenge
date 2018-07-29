#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-20
# Task: 
# Insights:
import re
import threading
from datetime import datetime

import h5py
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
import pandas as pd
import matplotlib.pyplot as plt
import os


def freeze_model(model, freeze_before_layer):
    if freeze_before_layer == "ALL":
        for l in model.layers:
            l.trainable = False
    else:
        freeze_before_layer_index = -1
        for i, l in enumerate(model.layers):
            if l.name == freeze_before_layer:
                freeze_before_layer_index = i
        for l in model.layers[:freeze_before_layer_index + 1]:
            l.trainable = False


def show_training_log(log_csv, fig_save_path=None, show_columns=None, epochs=None, xlim_range=None, ylim_range=None):
    assert isinstance(epochs, int) or epochs is None
    # cnames = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]
    cnames = ["b", "g", "r", "c", "m", "y", "k"]
    df = pd.read_csv(log_csv)
    columns = list(df.columns)
    assert len(columns) // 2 <= len(cnames)
    if show_columns:
        assert len(np.setdiff1d(show_columns, columns)) == 0
        train_columns = show_columns
        val_columns = ["val_" + x for x in train_columns]
    else:
        split = len(columns) // 2 + 1
        train_columns = columns[1:split]
        val_columns = columns[split:]
    # print(len(train_columns))
    # print(len(val_columns))
    x = list(df[columns[0]])[:epochs]
    for i in range(len(train_columns)):
        plt.plot(x, df[train_columns[i]][:epochs], cnames[i] + "--", label=train_columns[i])
        plt.plot(x, df[val_columns[i]][:epochs], cnames[i] + "-", label=val_columns[i])
    plt.xlabel("epochs")
    plt.ylabel("metrics")
    plt.legend()

    if xlim_range:
        plt.xlim(xlim_range[0], xlim_range[1])
    if ylim_range:
        plt.ylim(ylim_range[0], ylim_range[1])

    title = os.path.splitext(os.path.basename(log_csv))[0]
    plt.title(title)
    plt.grid()
    if fig_save_path:
        plt.savefig(fig_save_path)
    else:
        plt.show()


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def mean_iou_ch0(y_true, y_pred):
    y_true = y_true[..., 0]
    y_pred = y_pred[..., 0]
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def mean_iou_ch1(y_true, y_pred):
    y_true = y_true[..., 1]
    y_pred = y_pred[..., 1]
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# not used here
class DataGeneratorCustom(keras.utils.Sequence):
    def __init__(self, hdf5_path, batch_size, mode, shuffle=True):
        self.f = h5py.File(hdf5_path, 'r')
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle = shuffle
        assert self.mode in ['train', 'val']
        if self.mode == "train":
            keys = self.f["train_id"].value
        elif mode == "val":
            keys = self.f["val_id"].value
        else:
            # keys = self.f["test_id"].value
            raise NotImplementedError
        self.keys = [k.tostring().decode() for k in keys]
        self.gen = self.generator()
        self.is_lock = threading.Lock()

    # @thread_safe_generator
    def generator(self):
        steps = self.__len__()
        total_steps = 0
        while 1:
            if self.shuffle:
                np.random.shuffle(self.keys)
            for step in range(steps):
                total_steps += 1
                print("\ttotal step:", total_steps)
                print(threading.current_thread().name)
                tmp_keys = self.keys[step * self.batch_size: (step + 1) * self.batch_size]
                images, labels = get_input_data(self.f, tmp_keys, transform=False, is_train=True)
                images = preprocess(images)
                labels = preprocess(labels, mode="mask")
                yield images, labels

    def __len__(self):
        return int(np.ceil(len(self.keys) / float(self.batch_size)))

    def __getitem__(self, idx):
        with self.is_lock:
            batch_x, batch_y = next(self.gen)
            return np.asarray(batch_x), np.asarray(batch_y)


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
