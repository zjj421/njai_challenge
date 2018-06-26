#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-21
# Task:
# Insights:
import os
from datetime import datetime

from keras.models import load_model
import numpy as np

from matplotlib import pyplot as plt
from func.utils import DataGeneratorCustom, DataReader


def inference():
    model_path = "/home/topsky/helloworld/study/njai_challenge/cbct/func/model.h5"
    # model_path = "/home/topsky/helloworld/study/unet/unet_membrane.hdf5"
    h5_data_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data.hdf5"
    model = load_model(model_path)
    # model.summary()

    data_val = DataReader(h5_data_path, batch_size=1, mode="train", shuffle=False)
    x_val, y_val = data_val.images, data_val.labels


    # gen = DataGeneratorCustom(h5_data_path, batch_size=1, mode='train', shuffle=False).gen
    # images, labels = next(gen)
    outputs = model.predict_on_batch(np.expand_dims(x_val[8], axis=0))
    print(type(outputs))
    print(outputs.shape)
    image = np.squeeze(outputs, axis=(0, 3))
    print(image.shape)
    # image *= 255
    image[image > 0.48017228] = 1
    image[image <= 0.48017228] = 0
    print(image[0][:10])
    # image = x_val.flatten()
    # image = sorted(image, reverse=True)
    # print(image[:10])
    # print(image[-10:])
    # exit()
    # print(image)
    # print(image.shape)
    # image = np.squeeze(images, axis=(0, 3))
    # print(image)
    # print(image.shape)
    # 传入cmap='gray'指定图片为黑白

    plt.imshow(image, cmap='gray')
    plt.show()


def __main():
    np.set_printoptions(threshold=np.inf)
    inference()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
