#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-21
# Task:
# Insights:
import os
from datetime import datetime

import numpy as np
import scipy.misc
from PIL import Image
from keras.models import load_model
from matplotlib import pyplot as plt

from cbct.func.model_inception_resnet_v2 import get_inception_resnet_v2_unet_sigmoid
from cbct.func.utils import preprocess
from tqdm import tqdm

from module.utils_public import apply_mask, get_file_path_list


class ModelDeployment(object):
    def __init__(self, model_def, model_trained):
        try:
            model = load_model(model_trained)
        except:
            # print("Loading imagenet weights ...")
            model = model_def(weights=None)

            print("Loading weights ...")
            model.load_weights(model_trained)
        model.summary()
        self.model = model

    def read_images(self, image_path_lst):
        print("Loading images ...")
        imgs = []
        for image_path in tqdm(image_path_lst):
            img = Image.open(image_path)
            img = np.array(img)
            if len(img.shape) == 3:
                img = img[:, :, 0]
            img = np.expand_dims(img, -1)
            imgs.append(img)
        return np.array(imgs)

    def predict(self, images, batch_size):
        images = preprocess(images, mode="image")
        outputs = self.model.predict(images, batch_size)
        return outputs

    def predict_and_save_image_masks(self, image_path_lst, save_dir, batch_size=32, color=None, alpha=0.5):
        images = self.read_images(image_path_lst)
        masks = self.predict(images, batch_size)
        masks = np.where(masks > 0.5, 1, 0)

        images = [np.concatenate([image for i in range(3)], axis=-1) for image in images]
        masks = [mask[..., 0] for mask in masks]

        if color is None:
            color = [0, 191, 255]
        image_masks = [apply_mask(image, mask, color, alpha) for image, mask in zip(images, masks)]
        dst_image_path_lst = [os.path.join(save_dir, os.path.basename(x)) for x in image_path_lst]
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for i in range(len(image_masks)):
            scipy.misc.toimage(image_masks[i], cmin=0.0, cmax=...).save(dst_image_path_lst[i])

    # no test
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
    model_path = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/inception_resnet_v2_all_train_1.h5"
    image_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/train"
    pred_mask_image_save_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/pred_mask_images_20180714"
    image_path_lst = get_file_path_list(image_dir)
    model = ModelDeployment(get_inception_resnet_v2_unet_sigmoid, model_path)
    model.predict_and_save_image_masks(image_path_lst, pred_mask_image_save_dir, batch_size=4, color=None, alpha=0.5)


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
