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
from cbct.func.model_inception_resnet_v2 import get_inception_resnet_v2_unet_sigmoid
from cbct.func.utils import preprocess, de_preprocess
from func.data_io import DataSet, get_images
from keras.models import load_model
from matplotlib import pyplot as plt
from tqdm import tqdm
import keras.backend as K
from module.utils_public import apply_mask, get_file_path_list


class ModelDeployment(object):
    def __init__(self, model_def, model_trained):
        model = model_def

        print("Loading weights ...")
        model.load_weights(model_trained)
        model.summary()
        self.model = model

    def predict_and_save_stage1_masks(self, h5_data_path, h5_result_saved_path, fold_k=0, batch_size=32):
        """
        从h5data中读取images进行预测，并把预测mask保存进h5data中。
        Args:
            h5_data_path: str, 存放有训练数据的h5文件路径。
            batch_size: int, 批大小。

        Returns: None.

        """
        f_h5 = h5py.File(h5_data_path, 'r+')
        f_result = h5py.File(h5_result_saved_path, "a")
        images_grp = f_h5["images"]
        try:
            stage1_predict_masks_grp = f_result.create_group("stage1_fold{}_predict_masks".format(fold_k))
        except:
            stage1_predict_masks_grp = f_h5["stage1_fold{}_predict_masks".format(fold_k)]
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
        images = preprocess(images, mode="image")
        masks = self.predict(images, batch_size)
        print(masks.shape)
        masks = np.squeeze(masks, axis=-1)
        print("Saving predicted masks ...")
        for i, key in enumerate(keys):
            stage1_predict_masks_grp.create_dataset(key, dtype=np.float32, data=masks[i])
        print("Done.")

    def predict_and_show(self, images, output_channels=1):
        if isinstance(images, str):
            images_src = self.read_images([images])
        else:
            images_src = images
        images = preprocess(images_src, mode="image")
        predict_mask = self.predict(images, 1)
        result = np.squeeze(predict_mask, axis=0)
        if output_channels == 2:
            result0 = result[..., 0]
            result0 = preprocess(result0, mode="mask")
            result0 = np.where(result0 > 0, 255, 0)
            result1 = result[..., 1]
            result1 = preprocess(result1, mode="mask")
            result1 = np.where(result1 > 0, 255, 0)
            result = np.concatenate((np.squeeze(images_src, axis=[0, -1]), result0, result1), axis=1)
            plt.imshow(result, cmap="gray")
        else:
            plt.imshow(result, cmap="gray")


        plt.show()

    def read_images(self, image_path_lst):
        print("Loading images ...")
        imgs = []
        for image_path in tqdm(image_path_lst):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, -1)
            imgs.append(img)
        return np.array(imgs)

    def predict(self, images, batch_size):
        outputs = self.model.predict(images, batch_size)
        return outputs

    def predict_and_save_image_masks(self, image_path_lst, save_dir, batch_size=32, color=None, alpha=0.5):
        images = self.read_images(image_path_lst)
        images = preprocess(images, mode="image")
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


def inference_2stages(model_def_stage1, model_weights_stage1, model_def_stage2, model_weights_stage2, h5_data_path,
                      val_fold_nb, pred_save_dir, image_masks_save_dir):
    dataset = DataSet(h5_data_path, val_fold_nb)
    # keys = dataset.val_keys
    keys = dataset.train_keys
    images = get_images(dataset.f_h5, keys)
    if len(get_file_path_list(pred_save_dir, ".npy")) == 0:
        model = model_def_stage1
        print("Loading stage1 weights ...")
        model.load_weights(model_weights_stage1)
        # model.summary()

        stage1_outputs = model.predict(images, 4)
        print(stage1_outputs.shape)
        # K.clear_session()

        stage2_inputs = np.concatenate([images, stage1_outputs], axis=-1)
        model = model_def_stage2
        print("Loading stage2 weights ...")
        model.load_weights(model_weights_stage2)
        outputs = model.predict(stage2_inputs, 4)
        predicted_masks = np.squeeze(outputs, axis=-1)

        if not os.path.isdir(pred_save_dir):
            os.makedirs(pred_save_dir)
        np_save_path = os.path.join(pred_save_dir, "predicted_masks_2stages_fold{}".format(val_fold_nb) + ".npy")
        np.save(np_save_path, predicted_masks)
    else:
        np_save_path = os.path.join(pred_save_dir, "predicted_masks_2stages_fold{}".format(val_fold_nb) + ".npy")

        predicted_masks = np.load(np_save_path)

    masks = np.where(predicted_masks > 0.5, 1, 0)
    images = de_preprocess(images)
    images = [np.concatenate([image for i in range(3)], axis=-1) for image in images]

    color = [0, 191, 255]
    image_masks = [apply_mask(image, mask, color, alpha=0.5) for image, mask in zip(images, masks)]
    dst_image_path_lst = [os.path.join(image_masks_save_dir, "{:03}.tif".format(int(x))) for x in keys]
    if not os.path.isdir(image_masks_save_dir):
        os.makedirs(image_masks_save_dir)
    for i in range(len(image_masks)):
        cv2.imwrite(dst_image_path_lst[i], image_masks[i])


def do_infer_2stages():
    model_def_stage1 = get_inception_resnet_v2_unet_sigmoid(input_shape=(576, 576, 1), weights=None, output_channels=1)
    model_weights_stage1 = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/inception_resnet_v2_stage1_fold1.h5"
    model_def_stage2 = get_inception_resnet_v2_unet_sigmoid(input_shape=(576, 576, 2), weights=None, output_channels=1)
    model_weights_stage2 = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/inception_resnet_v2_stage2_fold1.h5"
    h5_data_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data_0717.hdf5"
    val_fold_nb = 1
    pred_save_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/pred_masks_2stage_fold1_train_0721"
    image_masks_save_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/pred_image_masks_2stage_fold1_train_0721"
    inference_2stages(model_def_stage1, model_weights_stage1, model_def_stage2, model_weights_stage2, h5_data_path,
                      val_fold_nb, pred_save_dir, image_masks_save_dir)


def infer_do():
    model_path = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/inception_resnet_v2_all_train_1.h5"
    # model_path = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/inception_resnet_v2_fold1_2channels.h5"
    # image_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/train"
    # pred_mask_image_save_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/pred_mask_stage1"
    # image_path_lst = get_file_path_list(image_dir)
    h5_data_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data_0717.hdf5"
    h5_predicted_masks_saved_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/predicted_masks_data_0717.hdf5"

    model_def = get_inception_resnet_v2_unet_sigmoid(weights=None, output_channels=2)
    model = ModelDeployment(model_def, model_path)

    dataset = DataSet(h5_data_path, val_fold_nb=1)
    keys = dataset.train_keys
    images = dataset.get_image_by_key(keys[1])

    model.predict_and_show(images, output_channels=2)
    # model.predict_and_save_stage1_masks(h5_data_path, h5_predicted_masks_saved_path, fold_k=2, batch_size=4)
    # model.predict_and_save_image_masks(image_path_lst, pred_mask_image_save_dir, batch_size=4, color=None, alpha=0.5)


def __main():
    np.set_printoptions(threshold=np.inf)
    # do_infer_2stages()
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
