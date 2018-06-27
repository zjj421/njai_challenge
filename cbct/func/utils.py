#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-20
# Task: 
# Insights:
import math
import threading
from datetime import datetime

import h5py
import keras
import numpy as np
from keras import backend as K

import tensorflow as tf

IMG_H, IMG_W, IMG_C = 576, 576, 1


def get_input_data(f_obj, tmp_keys, transform, is_train):
    images = []
    labels = []
    for key in tmp_keys:
        image = f_obj["images"][key].value
        if transform:
            image = transform(image)
        images.append(image)
        if is_train:
            label = f_obj["labels"][key].value
            labels.append(label)
    images = np.asarray(images)
    labels = np.asarray(labels)
    images = np.expand_dims(images, axis=-1)
    labels = np.expand_dims(labels, axis=-1)
    return images, labels


def preprocess(inputs_array, mode="images", input_channels=1):
    images = inputs_array / 127.5
    images -= 1.
    # images = inputs_array / 255
    if mode == "mask":
        images[images > 0.5] = 1
        images[images <= 0.5] = 0
    else:
        if input_channels == 3:
            images = np.concatenate([images for i in range(input_channels)], axis=-1)
            # print(images.shape)
    return images


# def get_input_data(f_obj, tmp_keys, transform, is_train):
#     images = []
#     labels = []
#     for key in tmp_keys:
#         image = f_obj["images"][key].value
#         if transform:
#             image = transform(image)
#         images.append(image)
#         if is_train:
#             label = f_obj["labels"][key].value
#             labels.append(label)
#     images = np.asarray(images)
#     labels = np.asarray(labels)
#     images = np.expand_dims(images, axis=-1)
#     labels = np.expand_dims(labels, axis=-1)
#     return images, labels


# def thread_safe_generator(f):
#     """A decorator that takes a generator function and makes it thread-safe.
#     https://github.com/fchollet/keras/issues/1638
#     http://anandology.com/blog/using-iterators-and-generators/
#     """
#
#     def g(*a, **kw):
#         return ThreadSafeIter(f(*a, **kw))
#
#     return g
#
#
# class ThreadSafeIter(object):
#     def __init__(self):
#         self.is_lock = threading.Lock()
#         # self.itt = itt
#
#     # def __next__(self, itt):
#     #     with self.is_lock:
#     #         return next(itt)
#
#     def __iter__(self):
#         return self
#
#     def next(self, itt):
#         with self.is_lock:
#             return next(itt)
#
#
# global safe_next
# safe_next = ThreadSafeIter()

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


def prepare_all_data(hdf5_data_path, mode):
    f = h5py.File(hdf5_data_path, 'r')
    assert mode in ['train', 'val']
    if mode == "train":
        keys = f["train_id"].value
    elif mode == "val":
        keys = f["val_id"].value
    else:
        # keys = self.f["test_id"].value
        raise NotImplementedError
    keys = [k.tostring().decode() for k in keys]
    images, labels = get_input_data(f, keys, transform=False, is_train=True)
    images = preprocess(images, mode="images", input_channels=1)
    labels = preprocess(labels, mode="mask")
    print(mode, "images.shape: {}".format(images.shape))
    print(mode, "labels.shape: {}".format(labels.shape))
    return images, labels


# deprecated
class DataReader(keras.utils.Sequence):
    def __init__(self, hdf5_data_path, batch_size, mode, shuffle=True):
        self.f = h5py.File(hdf5_data_path, 'r')
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
        self.is_lock = threading.Lock()
        self.images, self.labels = get_input_data(self.f, self.keys, transform=False, is_train=True)
        self.images = preprocess(self.images, mode="images", input_channels=1)
        self.labels = preprocess(self.labels, mode="mask")
        print(mode, "images.shape: {}".format(self.images.shape))
        print(mode, "labels.shape: {}".format(self.labels.shape))

    def __len__(self):
        return int(np.ceil(len(self.keys) / float(self.batch_size)))

    def __getitem__(self, idx):
        # 可能有Bug.
        with self.is_lock:
            batch_x, batch_y = self.images[idx], self.labels[idx]
            return np.asarray(batch_x), np.asarray(batch_y)


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
        # 可能有Bug.
        # if self.is_lock:

        with self.is_lock:
            batch_x, batch_y = next(self.gen)
            return np.asarray(batch_x), np.asarray(batch_y)


# h5py
def generate_data_custom(hdf5_path, batch_size, mode, shuffle=True):
    assert mode in ["train", "val"]
    # assert mode in ["train", "val", "test"]
    f = h5py.File(hdf5_path, mode="r")
    if mode == "train":
        keys = f["train_id"].value
    elif mode == "val":
        keys = f["val_id"].value
    else:
        keys = f["test_id"].value
    keys = [k.tostring().decode() for k in keys]
    # group_images = f["images"]
    # keys = list(group_images.keys())
    steps = int(math.ceil(len(keys) / batch_size))
    total_steps = 0
    while 1:
        if shuffle:
            np.random.shuffle(keys)
        for step in range(steps):
            total_steps += 1
            print("total step:", total_steps)
            # 最后一批数据比前面的少！
            tmp_keys = keys[step * batch_size: (step + 1) * batch_size]
            images, labels = get_input_data(f, tmp_keys, transform=False, is_train=True)
            yield images, labels


def __main():
    gen = DataGeneratorCustom()


# lmdb
# def get_input_data(lmdb_x, lmdb_y, tmp_keys, transform, is_train):
#     images = []
#     labels = []
#     for key in tmp_keys:
#         image = lmdb_x.read(key)
#         if transform:
#             image = transform(image)
#         images.append(image)
#         if is_train:
#             label = lmdb_y.read(key)
#             labels.append(label)
#     return images, labels
#
# def generate_data_custom(lmdb_x_path, lmdb_y_path, batch_size):
#     lmdb_x = LMDB(lmdb_x_path)
#     lmdb_y = LMDB(lmdb_y_path)
#     keys = lmdb_x.get_keys(n=lmdb_x.count())
#     print(keys)
#     steps = int(math.ceil(len(keys) / batch_size))
#     while True:
#         for step in range(steps):
#             tmp_keys = keys[step * batch_size: (step + 1) * batch_size]
#             images, labels = get_input_data(lmdb_x, lmdb_y, tmp_keys, transform=False, is_train=True)
#             print(len(images) == len(labels))
#             print(images[0].shape)
#             exit()
#             yield images, labels


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
