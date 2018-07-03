#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-21
# Task:
# Insights:
import os
from datetime import datetime

import numpy as np
from PIL import Image
from keras.models import load_model
from matplotlib import pyplot as plt

from cbct.func.model import unet_kaggle
from cbct.func.utils import preprocess
from module.utils_public import apply_mask


class ModelDeployment(object):
    def __init__(self, model_def, model_trained):
        try:
            model = load_model(model_trained)
        except:
            model = model_def()
            model.load_weights(model_trained)
        model.summary()
        self.model = model

    def _read_image(self, image_path):
        img = Image.open(image_path)
        img = np.array(img)
        if len(img.shape) == 3:
            img = img[:, :, 0]
        img = np.expand_dims(img, -1)
        return img

    def predict(self, image_path):
        img = self._read_image(image_path)
        input_ = img.copy()
        img = preprocess(img, mode="image")
        # print(np.squeeze(img, -1))
        img = np.expand_dims(img, axis=0)
        outputs = self.model.predict_on_batch(img)
        return input_, outputs[0]

    @staticmethod
    def show_image(**kw):
        assert "image" in kw.keys()
        mask = None
        image = kw["image"]
        assert len(image.shape) == 3, "Image should be 3-d array."
        if "mask" in kw.keys():
            mask = kw["mask"]
            assert len(mask.shape) == 2, "Mask should be 2-d array."
        if mask is not None:
            print("fff")
            color = [0, 191, 255]
            image = apply_mask(image, mask, color, alpha=0.5)
        plt.figure()
        plt.imshow(image)
        plt.show()


def infer_do():
    model_path = "/home/topsky/helloworld/study/njai_challenge/cbct/func/model.h5"
    model = ModelDeployment(unet_kaggle, model_path)
    image_path = "/media/topsky/HHH/jzhang_root/data/njai/cbct/train/002.tif"
    image, pred = model.predict(image_path)
    pred *= 255
    image = np.concatenate([image for i in range(3)], axis=-1)
    pred = np.squeeze(pred, axis=-1)
    # pred = np.concatenate([pred for i in range(3)], axis=-1)
    print(image.shape)
    print(pred.shape)
    kw = {"image": image, "mask": pred}
    model.show_image(**kw)


def __main():
    np.set_printoptions(threshold=np.inf)
    infer_do()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
