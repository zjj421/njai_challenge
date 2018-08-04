#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-15
# Task: 
# Insights:
import colorsys
import os
import random
import re
from datetime import datetime

import numpy as np





def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """
    Apply the given mask to the image.
    Args:
        image: 3-d numpy array, (h, w, c). RGB image, range from 0 to 255.
        mask: 2-d numpy array, (h, w). Grayscale image, range from 0 to 255.
        color: 3 item list, ie. [0, 191, 255].
        alpha: float, range from 0 to 1.

    Returns: 3-d numpy array.

    """
    assert (np.unique(mask) == np.array([0, 255])).all()
    mask_image = np.zeros_like(image)
    for c in range(3):
        mask_image[:, :, c] = np.where(mask == 255,
                                       image[..., c] * (1 - alpha) + alpha * color[c],
                                       image[..., c])
    return mask_image


def get_file_path_list(dir_path, ext=None):
    """
    从给定目录中获取所有文件的路径

    :param dir_path: 路径名
    :return: 该路径下的所有文件路径(path)列表
    """
    if ext:
        patt = re.compile(r".*{}$".format(ext))

    file_path_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if ext:
                result = patt.search(file)
                if not result:
                    continue
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
