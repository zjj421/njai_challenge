#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-30
# Task: 
# Insights: 
import os
from datetime import datetime

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from module.utils_public import apply_mask


def show_image():
    np.set_printoptions(threshold=np.nan)
    image_path = "/media/zj/share/data/njai_2018/cbct/train/001.tif"
    mask_path = "/media/zj/share/data/njai_2018/cbct/label/001.tif"
    # color = np.random.rand(3)
    color = [0, 191, 255]
    print(color)
    image = Image.open(image_path)
    image = np.array(image)
    mask = Image.open(mask_path)
    mask = np.array(mask)
    mask = mask[:, :, 0]
    print(image.shape)
    print(mask.shape)
    mask_image = apply_mask(image, mask, color, alpha=0.5)
    image = np.concatenate([image, mask_image], axis=1)
    plt.imshow(image)
    plt.show()
    # scipy.misc.toimage(image, cmin=0.0, cmax=...).save('outfile.jpg')


def data_analysis():
    np.set_printoptions(threshold=np.nan)
    data_root = "/media/zj/share/data/njai_2018/cbct"
    file_basename_lst = next(os.walk(os.path.join(data_root, "train")))[2]
    train_file_path_lst = [os.path.join(data_root, "train", x) for x in file_basename_lst]
    label_file_path_lst = [os.path.join(data_root, "label", x) for x in file_basename_lst]
    train_shape = []
    label_shape = []
    for i, file_path in enumerate(train_file_path_lst):
        im = Image.open(file_path)
        im = np.array(im, dtype=np.uint8)
        if len(im.shape) == 3:
            print(file_path)
            assert im[:, :, 0].all() == im[:, :, 1].all() and im[:, :, 0].all() == im[:, :, 2].all()
        train_shape.append(im.shape)
    for i, file_path in enumerate(label_file_path_lst):
        im = Image.open(file_path)
        im = np.array(im, dtype=np.uint8)
        label_shape.append(im.shape)
    print(set(train_shape))
    print(set(label_shape))
    # for i, s in enumerate(train_shape):
    #     if s == (576, 576, 3):
    #         print(train_file_path_lst[i])


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
