#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-26
# Task: 
# Insights:

from datetime import datetime

import numpy as np

from func.utils import DataReader


def check_baseline_accuracy():
    h5_data_path = "/home/jzhang/helloworld/mtcnn/cb/inputs/data.hdf5"
    data_val = DataReader(h5_data_path, batch_size=None, mode="train", shuffle=True)
    x_val, y_val = data_val.images, data_val.labels
    nb_correct = 0
    gts = y_val
    ests = np.zeros(gts.shape)
    print("hha")
    print(ests.shape)
    print(gts.shape)
    res = gts == ests
    res = res.flatten()
    print(res)
    for r in res:
        if r == True:
            nb_correct += 1
    print(len(res))
    print(nb_correct)
    print(nb_correct / len(res))

    # return acc / (num_batches * batch_size)


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
