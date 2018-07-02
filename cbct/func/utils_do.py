#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-26
# Task: 
# Insights:

from datetime import datetime

import numpy as np

from func.utils import prepare_all_data


def check_baseline_accuracy():
    h5_data_path = "/home/jzhang/helloworld/mtcnn/cb/inputs/data.hdf5"
    x, y = prepare_all_data(h5_data_path, mode="val")
    nb_correct = 0
    gts = y
    ests = np.zeros(gts.shape)
    res = gts == ests
    res = res.flatten()
    for r in res:
        if r == True:
            nb_correct += 1
    print("Base line acc: {} / {} = {}".format(nb_correct, len(res), nb_correct / len(res)))


def __main():
    check_baseline_accuracy()


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
