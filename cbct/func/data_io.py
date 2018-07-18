#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-7-17
# Task: 
# Insights:
import json
from datetime import datetime

import h5py
import numpy as np
from func.utils import preprocess


def get_masks(f_obj, tmp_keys, mask_nb):
    assert mask_nb in [0, 1]
    masks = []
    mask_str = "mask_{}".format(mask_nb)
    for key in tmp_keys:
        mask = f_obj[mask_str][key].value
        masks.append(mask)
    masks = np.array(masks)
    masks = np.expand_dims(masks, axis=-1)
    masks = preprocess(masks, mode="mask")
    return masks


def get_images(f_obj, tmp_keys):
    images = []
    for key in tmp_keys:
        image = f_obj["images"][key].value
        # 确保images只有一个通道，这里只取一个通道的信息。
        if len(image.shape) == 3:
            image = image[:, :, 0]
        images.append(image)
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)
    images = preprocess(images, mode="image")
    return images


def get_stage1_predict_masks(f_obj, tmp_keys):
    masks = []
    for key in tmp_keys:
        mask = f_obj["stage1_predict_masks"][key].value
        masks.append(mask)
    masks = np.array(masks)
    masks = np.expand_dims(masks, axis=-1)
    return masks


class DataSet(object):
    def __init__(self, h5_data_path, val_fold_nb):
        assert val_fold_nb in [0, 1, 2]
        f_h5 = h5py.File(h5_data_path, 'r')
        k_fold_map = json.loads(f_h5["k_fold_map"].value)
        folds = k_fold_map.keys()
        val_keys = []
        train_keys = []
        for key in folds:
            if key[0] == str(val_fold_nb):
                val_keys.extend(k_fold_map[key])
            else:
                train_keys.extend(k_fold_map[key])
        self.f_h5 = f_h5
        self.train_keys = train_keys
        self.val_keys = val_keys

    def prepare_stage1_data(self, mode):
        assert mode in ["train", "val"]
        if mode == "train":
            keys = self.train_keys
        else:
            keys = self.val_keys
        images = get_images(self.f_h5, keys)
        masks = get_masks(self.f_h5, keys, 1)
        return images, masks

    def prepare_stage2_data(self, mode):
        assert mode in ["train", "val"]
        if mode == "train":
            keys = self.train_keys
        else:
            keys = self.val_keys
        images = get_images(self.f_h5, keys)
        stage1_predict_masks = get_stage1_predict_masks(self.f_h5, keys)
        images = np.concatenate([images, stage1_predict_masks], axis=-1)
        masks = get_masks(self.f_h5, keys, 0)
        return images, masks


def __main():
    h5_data_path = "/home/jzhang/helloworld/mtcnn/cb/inputs/data_0717.hdf5"
    pass


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
