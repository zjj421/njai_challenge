#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-21
# Task:
# Insights:

from datetime import datetime

from keras.models import load_model
import numpy as np

from matplotlib import pyplot as plt
from cbct.func.utils import DataGeneratorCustom


def inference():
    model_path = "/home/topsky/helloworld/study/njai_challenge/cbct/func/model.h5"
    hdf5_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data.hdf5"
    model = load_model(model_path)
    model.summary()
    gen = DataGeneratorCustom(hdf5_path, batch_size=1, mode='train', shuffle=False).gen
    images, labels = next(gen)
    # outputs = model.predict_on_batch(images)
    # print(type(outputs))
    # print(outputs.shape)
    # image = np.squeeze(outputs, axis=(0, 3))
    # print(image)
    # print(image.shape)
    image = np.squeeze(labels, axis=(0, 3))
    print(image)
    print(image.shape)
    # 传入cmap='gray'指定图片为黑白
    # plt.imshow(image, cmap='gray')
    # plt.show()


def __main():
    np.set_printoptions(threshold=np.inf)
    inference()


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
