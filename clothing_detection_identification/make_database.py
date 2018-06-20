#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-15
# Task: make lmdb database
# Insights:
import h5py
from datetime import datetime
import pandas as pd
import numpy as np

from module.lmdb_io import LMDB
from module.utils_public import get_file_path_from_dir
import os
from PIL import Image


def make_database():
    data_root = "/media/topsky/HHH/jzhang_root/data/njai/端网云捷无人零售(衣物)图像识别比赛数据集"
    train_txt = "train.txt"
    lmdb_x_path = "train_lmdb_x"
    lmdb_y_path = "train_lmdb_y"
    lmdb_x = LMDB(lmdb_x_path)
    lmdb_y = LMDB(lmdb_y_path)
    with open(train_txt, "r") as f:
        lines = f.readlines()
        lines = [line.split() for line in lines[2:]]
        lines = np.array(lines)
    file_path_lst = [os.path.join(data_root, f) for f in lines[:, 0]]

    assert sorted(file_path_lst) == sorted(get_file_path_from_dir(data_root))
    labels = list(lines[:, 1:].astype(np.uint16))
    # keys = list(lines[:, 0])
    lmdb_y.write(labels, None, None, "labels")

    images = []
    for i, file_path in enumerate(file_path_lst):
        im = Image.open(file_path)
        im = np.array(im, dtype=np.uint8)
        images.append(im)
        if i % 100 == 0:
            print("loading images... ", i)
        if i % 10000 == 0:
            lmdb_x.write(images, None, None, "images")
            images.clear()
            print("images clear")
    lmdb_x.write(images, None, None, "images")
    print(lmdb_x.count())
    print(lmdb_y.count())


def make_hdf5_database():
    data_root = "/media/topsky/HHH/jzhang_root/data/njai/端网云捷无人零售(衣物)图像识别比赛数据集"
    train_txt = "train.txt"
    hdf5_path = "data.hdf5"
    f = h5py.File(hdf5_path, "w")
    x = f.create_group("images")
    y = f.create_group("labels")

    with open(train_txt, "r") as f:
        lines = f.readlines()
        lines = [line.split() for line in lines[2:]]
        lines = np.array(lines)
    file_path_lst = [os.path.join(data_root, f) for f in lines[:, 0]]

    assert sorted(file_path_lst) == sorted(get_file_path_from_dir(data_root))
    labels = list(lines[:, 1:].astype(np.uint16))
    for i, (file_path, label) in enumerate(zip(file_path_lst, labels)):
        idx = "{:08}".format(i + 1)
        im = Image.open(file_path)
        im = np.array(im, dtype=np.uint8)
        x.create_dataset(idx, dtype=np.uint8, data=im)  # [0, 255]
        y.create_dataset(idx, dtype=np.uint16, data=label)  # box坐标信息，用np.unit16表示，表示范围[0, 65535]。
        if i % 100 == 0:
            print("loading images and labels... ", i)


def __main():
    make_hdf5_database()


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
