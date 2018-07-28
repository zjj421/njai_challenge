#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-7-17
# Task: 
# Insights:
import json
from datetime import datetime

import h5py
import numpy as np


class DataSet(object):
    def __init__(self, h5_data_path, val_fold_nb):
        f_h5 = h5py.File(h5_data_path, 'r')
        k_fold_map = json.loads(f_h5["k_fold_map"].value)
        folds = k_fold_map.keys()
        val_keys = []
        train_keys = []
        for key in folds:
            if len(val_fold_nb) == 1:
                sub_key = key[0]
            else:
                sub_key = key
            if sub_key == val_fold_nb:
                val_keys.extend(k_fold_map[key])
            else:
                train_keys.extend(k_fold_map[key])
        self.f_h5 = f_h5
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.val_fold_nb = val_fold_nb

    def prepare_stage1_data(self, is_train):
        images = self.get_images(is_train)
        images = self.preprocess(images, mode="image")

        masks = self.get_masks(is_train, 1)
        masks = self.preprocess(masks, mode="mask")

        return images, masks

    def prepare_stage2_data(self, is_train):
        images = self.get_images(is_train)
        images = self.preprocess(images, mode="image")

        stage1_predict_masks = self.get_stage1_predict_masks(is_train)
        images = np.concatenate([images, stage1_predict_masks], axis=-1)
        masks = self.get_masks(is_train, 0)
        masks = self.preprocess(masks, mode="mask")

        return images, masks

    def prepare_1i_2o_data(self, is_train):
        images = self.get_images(is_train)
        images = self.preprocess(images, mode="image")

        masks_0 = self.get_masks(is_train, 0)
        masks_1 = self.get_masks(is_train, 1)
        masks = np.concatenate([masks_0, masks_1], axis=-1)
        masks = self.preprocess(masks, mode="mask")
        return images, masks

    def prepare_1i_1o_data(self, is_train, mask_nb=0):
        images = self.get_images(is_train)
        images = self.preprocess(images, mode="image")

        masks = self.get_masks(is_train, mask_nb)
        masks = self.preprocess(masks, mode="mask")

        return images, masks

    def get_image_by_key(self, key):
        image = self.f_h5["images"][key].value
        if len(image.shape) == 3:
            image = image[:, :, 0]
        return image

    def get_images(self, is_train):
        keys = self.get_keys(is_train)
        images = []
        for key in keys:
            image = self.f_h5["images"][key].value
            # 确保images只有一个通道，这里只取一个通道的信息。
            if len(image.shape) == 3:
                image = image[:, :, 0]
            images.append(image)
        images = np.array(images)
        images = np.expand_dims(images, axis=-1)
        return images

    def get_masks(self, is_train, mask_nb):
        assert mask_nb in [0, 1]
        keys = self.get_keys(is_train)
        masks = []
        mask_str = "mask_{}".format(mask_nb)
        for key in keys:
            mask = self.f_h5[mask_str][key].value
            masks.append(mask)
        masks = np.array(masks)
        masks = np.expand_dims(masks, axis=-1)
        return masks

    def get_stage1_predict_masks(self, is_train):
        f_h5 = h5py.File("/home/jzhang/helloworld/mtcnn/cb/inputs/predicted_masks_data.hdf5", "r")
        keys = self.get_keys(is_train)
        masks = []
        for key in keys:
            mask = f_h5["stage1_fold{}_predict_masks".format(self.val_fold_nb)][key].value
            masks.append(mask)
        masks = np.array(masks)
        masks = np.expand_dims(masks, axis=-1)
        return masks

    def get_keys(self, is_train):
        if is_train:
            keys = self.train_keys
        else:
            keys = self.val_keys
        return keys

    @staticmethod
    def de_preprocess(images, mode="mask"):
        if mode == "image":
            imgs = images + 1.
            imgs = imgs * 127.5
        else:
            imgs = np.where(images == 1, 255, 0)
        return imgs

    @staticmethod
    def preprocess(images, mode="mask"):
        assert mode in ["image", "mask"]
        if mode == "image":
            imgs = images / 127.5
            imgs -= 1.
        else:
            imgs = np.where(images > 0.5, 1, 0)
        return imgs


def __main():
    h5_data_path = ""
    pass


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
