#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-20
# Task: 
# Insights:
import math

import os
from datetime import datetime

import h5py
from PIL import Image
import numpy as np

from cbct.func.utils import generate_data_custom


def data_analysis():
    np.set_printoptions(threshold=np.nan)
    data_root = "/media/topsky/HHH/jzhang_root/data/njai/cbct"
    file_basename_lst = next(os.walk(os.path.join(data_root, "train")))[2]
    train_file_path_lst = [os.path.join(data_root, "train", x) for x in file_basename_lst]
    label_file_path_lst = [os.path.join(data_root, "label", x) for x in file_basename_lst]
    train_shape = []
    label_shape = []
    for i, file_path in enumerate(train_file_path_lst):
        im = Image.open(file_path)
        im = np.array(im, dtype=np.uint8)
        if len(im.shape) == 3:

            # print(im[:, :, 0])
            # print(im[:, :, 1])
            # print(im[:, :, 2])
            # print(im.shape)
            r = im[:, :, 0] == im[:, :, 1]
            r = r.flatten()
            for i in r:
                # print(i)
                if i == True:
                    print("dd")

            # print(r.shape)

            # print(im)
            exit()
        train_shape.append(im.shape)
    for i, file_path in enumerate(label_file_path_lst):
        im = Image.open(file_path)
        im = np.array(im, dtype=np.uint8)
        label_shape.append(im.shape)
    print(train_shape)
    print(label_shape)
    for i, s in enumerate(train_shape):
        if s == (576, 576, 3):
            print(train_file_path_lst[i])


# deprecated
# def make_database():
#     data_root = "/media/topsky/HHH/jzhang_root/data/njai/cbct"
#     lmdb_x_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/train_lmdb_x"
#     lmdb_y_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/train_lmdb_y"
#     lmdb_x = LMDB(lmdb_x_path)
#     lmdb_y = LMDB(lmdb_y_path)
#     file_basename_lst = next(os.walk(os.path.join(data_root, "train")))[2]
#     train_file_path_lst = [os.path.join(data_root, "train", x) for x in file_basename_lst]
#     label_file_path_lst = [os.path.join(data_root, "label", x) for x in file_basename_lst]
#     images = []
#     labels = []
#     # write images
#     for i, file_path in enumerate(train_file_path_lst):
#         im = Image.open(file_path)
#         im = np.array(im, dtype=np.uint8)
#         if len(im.shape) == 3:
#             im = im[:, :, 0]
#         im = np.expand_dims(im, axis=2)
#         images.append(im)
#         if i % 10 == 0 and i > 0:
#             print("loading images... ", i)
#         if i % 10 == 0 and i > 0:
#             lmdb_x.write(images, None, None, "images")
#             images.clear()
#             print("images clear")
#     lmdb_x.write(images, None, None, "images")
#     # write labels
#     print("-" * 100)
#     for i, file_path in enumerate(label_file_path_lst):
#         im = Image.open(file_path)
#         im = np.array(im, dtype=np.uint8)
#         if len(im.shape) == 3:
#             im = im[:, :, 0]
#         im = np.expand_dims(im, axis=2)
#         labels.append(im)
#         if i % 10 == 0 and i > 0:
#             print("loading images... ", i)
#         if i % 10 == 0 and i > 0:
#             lmdb_y.write(labels, None, None, "images")
#             labels.clear()
#             print("images clear")
#     lmdb_y.write(labels, None, None, "images")
#     print(lmdb_x.count())
#     print(lmdb_y.count())


def make_hdf5_database():
    data_root = "/media/zj/share/data/njai_2018/cbct"
    hdf5_path = "/home/zj/helloworld/study/njai_challenge/cbct/inputs/data_test.hdf5"
    f = h5py.File(hdf5_path, "w")
    grp_x = f.create_group("images")
    grp_y = f.create_group("labels")
    file_basename_lst = next(os.walk(os.path.join(data_root, "train")))[2]
    train_file_path_lst = [os.path.join(data_root, "train", x) for x in file_basename_lst]
    label_file_path_lst = [os.path.join(data_root, "label", x) for x in file_basename_lst]

    for i in range(len(file_basename_lst)):
        idx = '{:08}'.format(i + 1)
        image = Image.open(train_file_path_lst[i])
        image = np.array(image, dtype=np.uint8)
        if len(image.shape) == 3:
            image = image[:, :, 0]
        grp_x.create_dataset(idx, dtype=np.uint8, data=image)  # [0, 255]

        label = Image.open(label_file_path_lst[i])
        label = np.array(label, dtype=np.uint8)
        if len(label.shape) == 3:
            label = label[:, :, 0]
        grp_y.create_dataset(idx, dtype=np.uint8, data=label)
        if i % 10 == 0:
            print("loading images and labels... ", i)


def add_train_val_id_hdf5():
    hdf5_path = "/home/zj/helloworld/study/njai_challenge/cbct/inputs/data.hdf5"
    f = h5py.File(hdf5_path, mode="r+")
    # del f["train_id"], f["val_id"]
    # exit()
    keys = list(f["images"].keys())
    print(keys)
    print(type(keys[0]))
    keys = [x.encode() for x in keys]
    print(keys)
    print(type(keys[0]))
    keys = [np.void(x) for x in keys]
    print(keys)
    print(type(keys[0]))
    np.random.shuffle(keys)
    print(keys)
    split = int(len(keys) * 0.8)
    for name in f:
        print(name)
    f.create_dataset("train_id", data=keys[:split])
    f.create_dataset("val_id", data=keys[split:])
    v = f["train_id"].value
    v = [x.tostring().decode() for x in v]
    print(v)
    print(len(v))
    f.close()


def generate_data_custom_test():
    hdf5_path = "/home/zj/helloworld/study/njai_challenge/cbct/inputs/data.hdf5"
    # gen = Data_Generator(hdf5_path, 8)
    f = generate_data_custom(hdf5_path, 16, mode="train")
    print(len(f))
    print(type(f))
    for i in range(2):
        # print(len(images) == len(labels))
        # print(len(images))
        images, labels = next(f)
        print(len(images))
        print(images[0].shape)
        # print(labels[0].shape)


def __main():
    # make_hdf5_database()
    generate_data_custom_test()


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
