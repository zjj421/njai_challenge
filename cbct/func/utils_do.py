#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-20
# Task: 
# Insights:
import os
from datetime import datetime

from PIL import Image
import numpy as np

from cbct.func.utils import generate_data_custom
from module.lmdb_io import LMDB


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


def make_database():
    data_root = "/media/topsky/HHH/jzhang_root/data/njai/cbct"
    lmdb_x_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/train_lmdb_x"
    lmdb_y_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/train_lmdb_y"
    lmdb_x = LMDB(lmdb_x_path)
    lmdb_y = LMDB(lmdb_y_path)
    file_basename_lst = next(os.walk(os.path.join(data_root, "train")))[2]
    train_file_path_lst = [os.path.join(data_root, "train", x) for x in file_basename_lst]
    label_file_path_lst = [os.path.join(data_root, "label", x) for x in file_basename_lst]
    images = []
    labels = []
    # write images
    for i, file_path in enumerate(train_file_path_lst):
        im = Image.open(file_path)
        im = np.array(im, dtype=np.uint8)
        if len(im.shape) == 3:
            im = im[:, :, 0]
        im = np.expand_dims(im, axis=2)
        images.append(im)
        if i % 10 == 0 and i > 0:
            print("loading images... ", i)
        if i % 10 == 0 and i > 0:
            lmdb_x.write(images, None, None, "images")
            images.clear()
            print("images clear")
    lmdb_x.write(images, None, None, "images")
    # write labels
    print("-" * 100)
    for i, file_path in enumerate(label_file_path_lst):
        im = Image.open(file_path)
        im = np.array(im, dtype=np.uint8)
        if len(im.shape) == 3:
            im = im[:, :, 0]
        im = np.expand_dims(im, axis=2)
        labels.append(im)
        if i % 10 == 0 and i > 0:
            print("loading images... ", i)
        if i % 10 == 0 and i > 0:
            lmdb_y.write(labels, None, None, "images")
            labels.clear()
            print("images clear")
    lmdb_y.write(labels, None, None, "images")
    print(lmdb_x.count())
    print(lmdb_y.count())

def generate_data_custom_test():
    lmdb_x_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/train_lmdb_x"
    lmdb_y_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/train_lmdb_y"
    generate_data_custom(lmdb_x_path, lmdb_y_path, 8)

def __main():
    generate_data_custom_test()


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
