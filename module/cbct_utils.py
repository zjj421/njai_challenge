#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-7-28
# Task: 
# Insights: 

import os
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm






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
