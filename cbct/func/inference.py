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
from zf_unet_576_model import ZF_UNET_576


def inference():
    model_path = "/home/topsky/helloworld/study/njai_challenge/cbct/func/model.h5"
    # model_path = "/home/topsky/helloworld/study/unet/unet_membrane.hdf5"
    h5_data_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data.hdf5"
    # model_trained = "/home/jzhang/helloworld/mtcnn/cb/inputs/unet_576.h5"
    model = ZF_UNET_576()
    model.load_weights(model_path)
    # model = load_model(model_path)
    # model.summary()
    # exit()
    data_val = DataReader(h5_data_path, batch_size=1, mode="train", shuffle=False)
    x_val_src, y_val = data_val.images, data_val.labels

    x_val = np.concatenate([x_val_src for i in range(3)], axis=-1)
    idx = 6
    outputs = model.predict_on_batch(np.expand_dims(x_val[idx], axis=0))
    pred = np.squeeze(outputs, axis=(0, 3))
    # pred[pred > 0.48017228] = 1
    # pred[pred <= 0.48017228] = 0

    gt = np.squeeze(y_val[idx], axis=-1)
    image = x_val[idx]

    plt.figure()
    plt.axis("off")
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    # plt.axis("off")
    plt.title("train_image")
    plt.subplot(2, 2, 2)
    plt.imshow(gt, cmap='gray')
    # plt.axis("off")
    plt.title("gt")
    plt.subplot(2, 2, 3)
    plt.imshow(pred, cmap='gray')
    # plt.axis("off")
    plt.title("pred")
    plt.subplot(2, 2, 4)
    plt.title("src_image")
    plt.imshow(np.squeeze(x_val_src[idx], axis=-1), cmap='gray')


    # plt.imshow(pred, cmap='gray')
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
