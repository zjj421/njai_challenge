#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-20
# Task: 
# Insights:
import h5py
import math
from datetime import datetime

import keras
import numpy as np

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
    return images, labels

class data_generator_custom(keras.utils.Sequence):



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
    pass


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
