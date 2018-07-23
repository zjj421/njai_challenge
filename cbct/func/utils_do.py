#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-26
# Task: 
# Insights:

from datetime import datetime

import numpy as np
import os
from func.model_inception_resnet_v2 import get_inception_resnet_v2_unet_sigmoid
from func.train_model import get_learning_rate_scheduler
from func.utils import prepare_all_data, show_training_log
import keras.backend as K

def learning_rate_scheduler_test():
    epoch = 700
    current_lr = 0.01
    lr = get_learning_rate_scheduler(epoch, current_lr)
    print("dd")
    print(lr)


def do_show_training_log():
    log_csv = "/home/topsky/helloworld/study/njai_challenge/cbct/logs/log_inception_resnet_v2_gn_fold1_1i_1o_0.csv"
    show_training_log(log_csv, fig_save_path=None, show_columns=None, epochs=200)


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


def load_imagenet_weights_and_save(no):
    model_pretrained_weights = "/home/jzhang/helloworld/mtcnn/cb/model_weights/inception_resnet_v2_input1_output2_pretrained_weights.h5"
    model_def = get_inception_resnet_v2_unet_sigmoid(input_shape=(576, 576, 2), weights="imagenet", output_channels=1)
    # model_def.summary()
    model_def.save_weights(model_pretrained_weights)
    K.clear_session()



def __main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    do_show_training_log()
    # check_baseline_accuracy()
    # load_imagenet_weights_and_save()
    pass


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
