#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-21
# Task:
# Insights:
import h5py
import os
from datetime import datetime

import cv2
import numpy as np
import scipy.misc
from PIL import Image
from cbct.func.model_inception_resnet_v2 import get_inception_resnet_v2_unet_sigmoid_stage1
from cbct.func.utils import preprocess
from keras.models import load_model
from matplotlib import pyplot as plt
from tqdm import tqdm

from module.utils_public import apply_mask, get_file_path_list


class ModelDeployment(object):
    def __init__(self, model_def, model_trained):
        try:
            model = load_model(model_trained)
        except:
            # print("Loading imagenet weights ...")
            model = model_def

            print("Loading weights ...")
            model.load_weights(model_trained)
        model.summary()
        self.model = model

    def predict_and_save_stage1_masks(self, h5_data_path, batch_size=32):
        """
        从h5data中读取images进行预测，并把预测mask保存进h5data中。
        Args:
            h5_data_path: str, 存放有训练数据的h5文件路径。
            batch_size: int, 批大小。

        Returns: None.

        """
        f_h5 = h5py.File(h5_data_path, 'r+')
        images_grp = f_h5["images"]
        del f_h5["stage1_predict_masks"]
        try:
            stage1_predict_masks_grp = f_h5.create_group("stage1_predict_masks")
        except:
            stage1_predict_masks_grp = f_h5["stage1_predict_masks"]
        keys = images_grp.keys()
        images = []
        for key in keys:
            image = images_grp[key].value
            if len(image.shape) == 3:
                image = image[:, :, 0]
            images.append(image)
        images = np.array(images)
        images = np.expand_dims(images, axis=-1)
        print("Predicting ...")
        masks = self.predict(images, batch_size)
        print(masks.shape)
        masks = np.squeeze(masks, axis=-1)
        print("Saving predicted masks ...")
        for i, key in enumerate(keys):
            stage1_predict_masks_grp.create_dataset(key, dtype=np.float32, data=masks[i])
        print("Done.")

    def read_images(self, image_path_lst):
        print("Loading images ...")
        imgs = []
        for image_path in tqdm(image_path_lst):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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
    # model_path = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/inception_resnet_v2_all_train_1.h5"
    model_path = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/inception_v4_stage1.h5"
    # image_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/train"
    # pred_mask_image_save_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/pred_mask_stage1"
    # image_path_lst = get_file_path_list(image_dir)
    h5_data_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data_0717.hdf5"

    model_def = get_inception_resnet_v2_unet_sigmoid_stage1(weights=None)
    model = ModelDeployment(model_def, model_path)
    model.predict_and_save_stage1_masks(h5_data_path, batch_size=4)
    # model.predict_and_save_image_masks(image_path_lst, pred_mask_image_save_dir, batch_size=4, color=None, alpha=0.5)


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
