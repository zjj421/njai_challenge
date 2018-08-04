#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-7-28
# Task: 
# Insights: 

import os
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from module.competition_utils import IMAGE_FILE_LIST, MASK_FILE_LIST
from module.utils_public import apply_mask
import scipy.misc

def combine_image_mask(image_dir, mask_dir, image_mask_save_dir, is_test=True, color=None):
    if not os.path.isdir(image_mask_save_dir):
        os.makedirs(image_mask_save_dir)
    if color is not None:
        color = color
    else:
        color = [255, 106, 106]
    if is_test:
        image_path_lst = [os.path.join(image_dir, x) for x in IMAGE_FILE_LIST]
        mask_path_lst = [os.path.join(mask_dir, x) for x in MASK_FILE_LIST]
        result_path_slt = [os.path.join(image_mask_save_dir, x) for x in MASK_FILE_LIST]
    else:
        file_lst = next(os.walk(image_dir))[2]
        image_path_lst = [os.path.join(image_dir, x) for x in file_lst]
        mask_path_lst = [os.path.join(mask_dir, x) for x in file_lst]
        result_path_slt = [os.path.join(image_mask_save_dir, x) for x in file_lst]
    for i in tqdm(range(len(image_path_lst))):
        image = cv2.imread(image_path_lst[i], cv2.IMREAD_COLOR)[..., ::-1]
        mask = cv2.imread(mask_path_lst[i], cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask == 0, 0, 255)


        image_mask = apply_mask(image, mask, color=color)

        mask = np.expand_dims(mask, axis=-1)
        mask = np.concatenate([mask for i in range(3)], axis=-1)
        combined_image = np.concatenate([image, image_mask, mask], axis=1)[..., ::-1]
        cv2.imwrite(result_path_slt[i], combined_image)

def concatenate_same_name_image(dir_lst, result_save_dir):
    """
    将多个目录下的图片对应concatenate后保存。
    Args:
        dir_lst: list, 每个元素都是一个目录路径，每个目录下的文件名相同。
        result_save_dir:　str, 结果保存的路径。

    Returns:

    """
    if not os.path.isdir(result_save_dir):
        os.makedirs(result_save_dir)
    file_path_lst = []
    for i, dir in enumerate(dir_lst):
        if i == 0:
            # 保持文件顺序一致。
            sub_file_basename_lst = next(os.walk(dir))[2]
            sub_file_path_lst = [os.path.join(dir, x) for x in sub_file_basename_lst]
        else:
            sub_file_path_lst = [os.path.join(dir, x) for x in sub_file_basename_lst]
        file_path_lst.append(sub_file_path_lst)
    for i in tqdm(range(len(file_path_lst[0]))):
        img = cv2.imread(file_path_lst[0][i], cv2.IMREAD_GRAYSCALE)
        for j in range(1, len(file_path_lst)):
            img_next = cv2.imread(file_path_lst[j][i], cv2.IMREAD_GRAYSCALE)
            img = np.concatenate([img, img_next], axis=1)
        cv2.imwrite(os.path.join(result_save_dir, sub_file_basename_lst[i]), img)
    print("Done.")


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
