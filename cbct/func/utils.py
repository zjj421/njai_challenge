#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-20
# Task: 
# Insights:
import threading
from datetime import datetime

import h5py
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K

IMG_H, IMG_W, IMG_C = 576, 576, 1


def get_input_data(f_obj, tmp_keys, transform, is_train):
    images = []
    labels = []
    for key in tmp_keys:
        image = f_obj["images"][key].value
        # 确保images只有一个通道，这里只取一个通道的信息。
        if len(image.shape) == 3:
            image = image[:, :, 0]
        if transform:
            # label也需要做同样的transform。还没有实现。
            image = transform(image, seed=1)
        images.append(image)
        if is_train:
            # label_1 = f_obj["mask_1"][key].value
            label_2 = f_obj["mask_2"][key].value
            labels.append(label_2)
            # label = np.concatenate([label_1, label_2], 2)
            # labels.append(label)
    images = np.asarray(images)
    images = np.expand_dims(images, axis=-1)
    if is_train:
        labels = np.asarray(labels)
        return images, labels
    else:
        return images


def preprocess(inputs_array, mode="mask"):

    images = inputs_array / 127.5
    images -= 1.
    # images = inputs_array / 255
    if mode == "mask":
        images[images > 0.5] = 1
        images[images <= 0.5] = 0
    return images


def prepare_all_data(h5_data_path, mode):
    f = h5py.File(h5_data_path, 'r')
    assert mode in ['train', 'val']
    if mode == "train":
        keys = f["train_id"].value
    elif mode == "val":
        keys = f["val_id"].value
    else:
        # keys = self.f["test_id"].value
        raise NotImplementedError
    keys = [k.tostring().decode() for k in keys]
    images, labels = get_input_data(f, keys, transform=None, is_train=True)
    images = preprocess(images, mode="images")
    labels = preprocess(labels, mode="mask")
    print(mode, "images.shape: {}".format(images.shape))
    print(mode, "labels.shape: {}".format(labels.shape))
    return images, labels


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
