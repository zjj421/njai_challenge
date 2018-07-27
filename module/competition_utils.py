#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-7-23
# Task: 
# Insights: 

import json
import os
from datetime import datetime

import keras.backend as K
import numpy as np
import tensorflow as tf
from PIL import Image
from collections import Counter

"""
示例json_file
[
{"name": "03+261mask.tif", 
"size": [3, 3], 
"pixel": [1, 1, 1, 
		1, 1, 1, 
		1, 1, 1]},
{"name": "03+262mask.tif", 
"size": [2, 2], 
"pixel": [2, 2, 
		2, 2]}
]
"""


# 提交测试结果为与测试图片大小相同的（单通道）图片，图中仅包含被识别出的牙齿（像素值为255）和背景（像素值为0）。

### 选手提交的文件名称，对应答案集合5个tif文件
# file_list = ['03+261mask.tif', '03+262mask.tif', '04+246mask.tif', '04+248mask.tif', '04+251mask.tif']


def convert_submission(file_path_list, dst_file_path):
    file_list = [os.path.basename(x) for x in file_path_list]
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
            data['name'] = file_list[i]  # 定义dict的key为name，对应的value存入图片的名称
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
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    return K.mean(K.equal(y_true.flatten(), K.round(y_pred.flatten())), axis=-1)


def ensemble(pred_array, threshold=0.5):
    """
    mask ensemble.
    :param pred_array: 3-d numpy array, such as (h, w, c), each channel is a 2-d mask.
    :threshold: float, if the pixel value > threshold, then the value will convert to 255, else 0.
    :return: a single mask.
    """
    masks = np.asarray(pred_array)
    masks = np.where(masks > threshold,
                     255,
                     0)
    h, w = masks.shape[:2]
    ensemble_mask = np.zeros(shape=(h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            pixel_wise_list = masks[i, j, :]
            pixel_ensemble = Counter(pixel_wise_list).most_common(1)
            ensemble_mask[i, j] = pixel_ensemble[0][0]
    return ensemble_mask


def test_ensemble():
    pass


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
    test_ensemble()
    pass


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
