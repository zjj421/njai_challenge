#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-15
# Task: make lmdb database
# Insights: 

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
    images = []
    for i, file_path in enumerate(file_path_lst):
        if i > 999:
            break
        im = Image.open(file_path)
        im = np.array(im, dtype=np.uint8)
        images.append(im)
        if i % 100 == 0:
            print("loading images... ", i)
    labels = list(lines[:, 1:].astype(np.uint16))[:1000]

    keys = list(lines[:, 0])[:1000]
    lmdb_x.write(images, None, keys, "images")
    lmdb_y.write(labels, None, keys, "labels")
    print(lmdb_x.count())
    print(lmdb_y.count())


def __main():
    make_database()


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
