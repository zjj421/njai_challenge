#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-30
# Task: 
# Insights:
import h5py
import json
import os
from datetime import datetime
from pprint import pprint

import cv2
import numpy as np
from PIL import Image
from func.utils import show_training_log
from matplotlib import pyplot as plt

from module.utils_public import apply_mask


def do_show_training_log():
    log_csv = "/home/topsky/helloworld/study/njai_challenge/cbct/logs/log_inception_resnet_v2_gn_fold1_1i_2o_0724_0.csv"
    show_training_log(log_csv, fig_save_path=None, show_columns=["binary_acc_ch0"], epochs=300)


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


def cv_imread_test():
    np.set_printoptions(threshold=np.inf)
    image_path = "/media/topsky/HHH/jzhang_root/data/njai/cbct/label/051.tif"
    # img = cv2.imread(image_path)
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print(img.shape)
    # img = np.stack([img, img], axis=2)
    # print(img.shape)
    # print(img[0])


def k_fold_test():
    k_fold_path = "/home/topsky/helloworld/study/njai_challenge/module/k_fold.csv"
    with open(k_fold_path, 'r') as f:
        lines = f.readlines()
    k_fold_map = {}
    for line in lines:
        value, idx = line.split(",")
        value_start_end = value.strip().split("-")
        value_lst = ["{:03}".format(x) for x in range(int(value_start_end[0]), int(value_start_end[1]) + 1)]
        idx = idx.strip()
        if idx not in k_fold_map.keys():
            k_fold_map[idx] = value_lst
        else:
            k_fold_map[idx].extend(value_lst)
    # with open("k_fold_map.json", "w") as f:
    #     json.dump(k_fold_map, f)
    for i in k_fold_map.keys():
        print(i, len(k_fold_map[i]))


def read_h5_test():
    h5_data_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data_0717.hdf5"
    f_h5 = h5py.File(h5_data_path, 'r+')
    stage1_predict_mask_grp = f_h5["stage1_predict_masks"]
    for i, key in enumerate(stage1_predict_mask_grp.keys()):
        mask = stage1_predict_mask_grp[key].value
        print(mask.max())
        print(mask)
        print(mask.shape)
        exit()


def read_image_test():
    img_path = "/media/zj/share/data/njai_2018/cbct/train/001.tif"
    img1 = Image.open(img_path).convert("L")
    img1 = np.array(img1)
    print(img1.shape)
    img2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print(img2.shape)
    print(img1.all() == img2.all())


def __main():
    np.set_printoptions(threshold=np.inf)
    # read_h5_test()
    do_show_training_log()
    pass


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
