#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-7-23
# Task: 
# Insights: 

from datetime import datetime

import os
from PIL import Image
import json
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.applications import ResNet50


def convert_submission(file_path_list, dst_file_path):
    if not os.path.isdir(os.path.dirname(dst_file_path)):
        os.makedirs(os.path.dirname(dst_file_path))
    res = []
    with open(dst_file_path, "w") as f:
        for i in range(0, len(file_path_list)):
            tmp_pic = Image.open(file_path_list[i]).convert('L')  # 读取图片，按照要求转化为灰度图
            (x, y) = tmp_pic.size  # 找到图片size
            tmp_pixel = []  # 创建一个list，便于记录pixel信息
            data = {}  # 创建一个dict，记录每张图片中，要被写入的信息
            for p in range(x):
                for q in range(y):
                    tmp_pixel.append(tmp_pic.getpixel((p, q)))  # 遍历每个像素点，把pixel信息存入tmp_pixel
            data['name'] = file_path_list[i]  # 定义dict的key为name，对应的value存入图片的名称
            data['size'] = [x, y]  # 定义dict的key为size，对应的value存入图片的size
            data['pixel'] = tmp_pixel  # 定义dict的key为pixel，对应的value存入图片的pixel
            res.append(data)
        f.write(json.dumps(res))
    print("All have done.")


def get_pixel_wise_acc(y_true, y_pred):
    """

    Args:
        y_true: 2-d numpy array, such as (576, 576).
        y_pred: same as y_true.

    Returns: float.

    """
    return K.mean(K.equal(y_true.flatten(), K.round(y_pred.flatten())), axis=-1)







def test_binary_acc():
    y_true = np.arange(16).reshape(4, 4)
    # y_true = np.expand_dims(y_true, axis=2)
    # y_true = np.repeat(y_true, repeats=2, axis=-1)
    print(y_true.shape)
    y_pred = np.arange(16).reshape(4, 4)
    y_pred[0][0] = 101
    # y_pred[0][1] = 101

    # y_pred = np.expand_dims(y_pred, axis=2)
    # y_pred = np.repeat(y_pred, repeats=2, axis=-1)
    # y_pred[0][0][1] = 0


    # print(y_true)
    print(y_pred)
    acc = get_pixel_wise_acc(y_true, y_pred)
    with tf.Session() as sess:
        print(sess.run(acc))


def __main():
    test_binary_acc()
    pass


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
