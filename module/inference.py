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

from cbct.func.model import unet_kaggle
from cbct.func.utils import prepare_all_data
from cbct.zf_unet_576_model import dice_coef_loss, dice_coef


def inference():
    model_path = "/home/topsky/helloworld/study/njai_challenge/cbct/func/model.h5"
    # model_path = "/home/topsky/helloworld/study/unet/unet_membrane.hdf5"
    h5_data_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data.hdf5"
    model = unet_kaggle()
    model.load_weights(model_path)
    # model = load_model(model_path)
    model.compile(optimizer="adam", loss=dice_coef_loss, metrics=["acc", dice_coef])
    model.summary()
    x_val, y_val = prepare_all_data(h5_data_path, mode="val")
    print(x_val.shape, y_val)

    idx = 6
    outputs = model.predict_on_batch(np.expand_dims(x_val[idx], axis=0))
    pred = np.squeeze(outputs, axis=0)
    mask_1 = pred[:, :, 0]
    mask_2 = pred[:, :, 1]
    gt = y_val[idx][0]
    image = x_val[idx]

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    plt.title("train_image")

    plt.subplot(2, 2, 2)
    plt.imshow(gt, cmap='gray')
    plt.axis("off")
    plt.title("gt")

    plt.subplot(2, 2, 3)
    plt.imshow(mask_1, cmap='gray')
    plt.axis("off")
    plt.title("mask_1")

    plt.subplot(2, 2, 4)
    plt.imshow(mask_2, cmap='gray')
    plt.axis("off")
    plt.title("mask_2")

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
