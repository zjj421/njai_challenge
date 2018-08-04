#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-8-3
# Task: 
# Insights: 

from datetime import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt
from module.utils_public import apply_mask


def aug_test():
    img = "/media/topsky/HHH/jzhang_root/data/njai/cbct/train/087.tif"
    mask = "/media/topsky/HHH/jzhang_root/data/njai/cbct/label/087.tif"
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask, cv2.IMREAD_UNCHANGED)
    mask = np.where(mask == 0, 0, 255)
    # print(np.unique(img))
    # print(np.unique(mask))
    # exit()

    # if len(img.shape) == 2:
    #     img = np.concatenate([np.expand_dims(img, axis=-1) for i in range(3)], axis=-1)
    image_mask = apply_mask(img, mask, color=[254, 106, 106])
    plt.imshow(image_mask)
    plt.show()
    print(img.shape)
    print(mask.shape)

def __main():
    aug_test()
    pass


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")