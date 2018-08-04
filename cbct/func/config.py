#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-7-17
# Task: 
# Insights:
import os
from datetime import datetime

class Config(object):
    def __init__(self):
        self.img_h, self.img_w, self.img_c = 576, 576, 1
        self.log_root = "/home/jzhang/helloworld/mtcnn/cb/logs"
        # if not os.path.isdir(self.log_root):
        #     os.makedirs(self.log_root)
        self.learning_rate_scheduler_csv = "/home/jzhang/helloworld/mtcnn/cb/func/learning_rate_scheduler.csv"

        self.input_channels = 1
        self.output_channels = 2
        self.random_crop_size = None
        self.mask_nb = 0
        self.model_weights_save_root = "/home/jzhang/helloworld/mtcnn/cb/model_weights/new"
        self.model_name = "se_densenet_gn"
        # self.model_name = "se_inception_resnet_v2_gn"


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
