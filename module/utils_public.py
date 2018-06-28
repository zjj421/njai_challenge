#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-15
# Task: 
# Insights:
import os
import numpy as np
from datetime import datetime


def apply_mask(image, mask, color, alpha=0.5):
    """
    Apply the given mask to the image.
    """
    mask_image = image.copy()
    for c in range(3):
        mask_image[:, :, c] = np.where(mask == 255,
                                       mask_image[:, :, c] * (1 - alpha) + alpha * color[c],
                                       mask_image[:, :, c])
    return mask_image


def get_file_path_from_dir(dir_path):
    """
    从给定目录中获取所有文件的路径

    :param dir_path: 路径名
    :return: 该路径下的所有文件路径(path)列表
    """
    file_path_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            path = os.path.join(root, file)
            file_path_list.append(path)
    print("'{}'目录中文件个数 : {}".format(os.path.basename(dir_path), len(file_path_list)))
    return file_path_list


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
