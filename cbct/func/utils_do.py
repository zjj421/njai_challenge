#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-26
# Task: 
# Insights:

from datetime import datetime

import numpy as np

from func.train_model import get_learning_rate_scheduler
from func.utils import prepare_all_data, show_training_log


def learning_rate_scheduler_test():
    epoch = 700
    current_lr = 0.01
    lr = get_learning_rate_scheduler(epoch, current_lr)
    print("dd")
    print(lr)


def do_show_training_log():
    log_csv = "/home/jzhang/helloworld/mtcnn/cb/logs/inception_v4_stage1_stage1.csv"
    show_training_log(log_csv)


def check_baseline_accuracy():
    h5_data_path = "/home/jzhang/helloworld/mtcnn/cb/inputs/data.hdf5"
    x, y = prepare_all_data(h5_data_path, mode="train")
    nb_correct = 0
    gts = y
    ests = np.zeros(gts.shape)
    res = gts == ests
    res = res.flatten()
    for r in res:
        # can not use 'r is True' here.
        if r == True:
            nb_correct += 1
    # 10 samples, val acc: 0.9489459755979939
    # 90 samples, train acc: 0.9565696266289437
    print("Base line acc: {} / {} = {}".format(nb_correct, len(res), nb_correct / len(res)))


def __main():
    # check_baseline_accuracy()
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
