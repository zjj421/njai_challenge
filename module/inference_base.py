#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-21
# Task:
# Insights:

import os
from datetime import datetime

import cv2
import h5py
import keras.backend as K
import numpy as np
import tensorflow as tf
from cbct.func.model_inception_resnet_v2 import sigmoid_dice_loss, binary_acc_ch0
from func.config import Config
from func.data_io import DataSet
from func.model_se_inception_resnet_v2_gn import get_se_inception_resnet_v2_unet_sigmoid_gn
from matplotlib import pyplot as plt
from tqdm import tqdm

from module.augmentation import tta_predict, hflip_images, rotate_images
from module.competition_utils import get_pixel_wise_acc
from module.utils_public import apply_mask, get_file_path_list

CONFIG = Config()


def inference_from_files():
    model_def = get_se_inception_resnet_v2_unet_sigmoid_gn(weights=None, output_channels=2)
    model_weights = ""


class ModelDeployment(object):
    def __init__(self, model_def, model_weights, show_model=False):
        model = model_def

        print("Loading weights ...")
        model.load_weights(model_weights)

        self.model = model
        if show_model:
            self.model.summary()

    def reload_weights(self, model_weights):
        print("Reloading weights ...")
        self.model.load_weights(model_weights)

    def evaluate(self, x_val, y_val):
        x_val = DataSet.preprocess(x_val, "image")
        y_val = DataSet.preprocess(y_val, "mask")
        fit_loss = sigmoid_dice_loss
        fit_metrics = [binary_acc_ch0]
        self.model.compile(loss=fit_loss,
                           optimizer="Adam",
                           metrics=fit_metrics)
        # Score trained model.
        scores = self.model.evaluate(x_val, y_val, batch_size=5, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    @staticmethod
    def postprocess(masks):
        """
        将预测结果转化为二值(0 or 1)矩阵。
        Args:
            masks: numpy array

        Returns:

        """
        return np.where(masks > 0.5, 1, 0)

    def read_images(self, image_path_lst):
        """

        Args:
            image_path_lst: list, 每个元素表示图片的路径。

        Returns: 4-d numpy array image. (b, h, w, c=1).

        """
        print("Loading images ...")
        imgs = []
        for image_path in tqdm(image_path_lst):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, -1)
            imgs.append(img)
        return np.array(imgs)

    def predict(self, images, batch_size, use_channels=2):
        """
        对未预处理过的图片进行预测。
        Args:
            images: 4-d numpy array. preprocessed image. (b, h, w, c=1)
            batch_size:
            use_channels: int, default to 1. 如果模型输出通道数为2，可以控制输出几个channel.默认输出第一个channel的预测值.

        Returns: numpy array.

        """
        images = DataSet.preprocess(images, mode="image")
        outputs = self.model.predict(images, batch_size)
        if use_channels == 1:
            outputs = outputs[..., 0]
        return outputs

    def tta_predict(self, images, batch_size=1, use_channels=2):
        """
        对未预处理过的图片进行tta预测。
        :param images: 4-d numpy array. (b, h, w, c)
        :return: 4-d numpy array, (b, h, w, c)
        """
        rotate_angle_lst = [0, 4, 8, 12]
        h_flip_lst = [0, 1]
        bs, height, width, channels = images.shape
        aug_lst = [(is_h_flip, rotate_angle) for is_h_flip in h_flip_lst
                   for rotate_angle in rotate_angle_lst]
        preds = []
        for aug in tqdm(aug_lst):
            imgs = images.copy()
            imgs = hflip_images(imgs, aug[0])
            imgs = rotate_images(imgs, aug[1])
            pred = self.predict(imgs, batch_size=batch_size, use_channels=use_channels)
            pred = rotate_images(pred, -aug[1])
            pred = hflip_images(pred, aug[0])
            preds.append(pred)
        preds = np.array(preds)  # (nb_aug, b, h, w, c)

        preds_0 = preds[..., 0]  # (nb_aug, b, h, w)
        pred_0 = np.zeros(shape=(bs, height, width), dtype=np.float64)
        for b in range(bs):
            pred_b_0 = preds_0[:, b, :, :]  # (nb_aug, h, w)
            pred_b_0 = np.mean(pred_b_0, axis=0)  # (h, w)
            pred_0[b] = pred_b_0
        pred_0 = np.expand_dims(pred_0, axis=-1)
        if use_channels == 1:
            return pred_0
        else:
            preds_1 = preds[..., 1]  # (nb_aug, b, h, w)
            pred_1 = np.zeros(shape=(bs, height, width), dtype=np.float64)
            for b in range(bs):
                pred_b_1 = preds_1[:, b, :, :]  # (nb_aug, h, w)
                pred_b_1 = np.mean(pred_b_1, axis=0)  # (h, w)
                pred_1[b] = pred_b_1
            pred_1 = np.expand_dims(pred_1, axis=-1)
            pred = np.concatenate([pred_0, pred_1], axis=-1)
            return pred

    def predict_from_files(self, image_path_lst, batch_size=5, use_channels=2, mask_file_lst=None, tta=False,
                           is_save_npy=False, is_save_mask0=False, is_save_mask1=False, result_save_dir=""):
        """
        给定图片路径列表，返回预测结果（未处理过的），如果指定了预测结果保存路径，则保存预测结果（已处理过的）。
        如果指定了预测结果保存的文件名列表，则该列表顺序必须与image_path_lst一致；
        如果没有指定预测结果保存的文件名列表，则自动生成和输入相同的文件名列表。
        Args:
            image_path_lst: list.
            batch_size:
            use_channels: 输出几个channel。
            mask_file_lst: list, 预测结果保存的文件名列表。
            tta: bool, 预测时是否进行数据增强。
            is_save_npy: bool, 是否保存npy文件。
            is_save_mask0: bool
            is_save_mask1: bool
            result_save_dir: str, 结果保存的目录路径。
        Returns: 4-d numpy array, predicted result.

        """
        imgs = self.read_images(image_path_lst)
        if tta:
            pred = self.tta_predict(imgs, batch_size=batch_size, use_channels=use_channels)
        else:
            pred = self.predict(imgs, batch_size=batch_size, use_channels=use_channels)
        if mask_file_lst is None:
            mask_file_lst = [os.path.basename(x) for x in image_path_lst]
        if is_save_npy:
            # 保存npy文件
            npy_dir = os.path.join(result_save_dir, "npy")
            self.save_npy(pred, mask_file_lst, npy_dir)
        if is_save_mask0:
            mask_nb = 0
            mask_save_dir = os.path.join(result_save_dir, "mask{}".format(mask_nb))
            self.save_mask(pred, mask_file_lst, mask_nb=mask_nb, result_save_dir=mask_save_dir)
        if is_save_mask1:
            mask_nb = 1
            mask_save_dir = os.path.join(result_save_dir, "mask{}".format(mask_nb))
            self.save_mask(pred, mask_file_lst, mask_nb=mask_nb, result_save_dir=mask_save_dir)
        return pred

    def predict_old(self, images, batch_size, use_channels=2):
        """
        对预处理过的图片进行预测。
        Args:
            images: 4-d numpy array. preprocessed image. (b, h, w, c=1)
            batch_size:
            use_channels: int, default to 1. 如果模型输出通道数为2，可以控制输出几个channel.默认输出第一个channel的预测值.

        Returns: numpy array.

        """
        outputs = self.model.predict(images, batch_size)
        if use_channels == 1:
            outputs = outputs[..., 0]
        return outputs

    def predict_from_files_old(self, image_path_lst, batch_size=5, use_channels=2, mask_file_lst=None, tta=False,
                               is_save_npy=False, is_save_mask0=False, is_save_mask1=False, result_save_dir=""):
        """
        给定图片路径列表，返回预测结果（未处理过的），如果指定了预测结果保存路径，则保存预测结果（已处理过的）。
        如果指定了预测结果保存的文件名列表，则该列表顺序必须与image_path_lst一致；
        如果没有指定预测结果保存的文件名列表，则自动生成和输入相同的文件名列表。
        Args:
            image_path_lst: list.
            batch_size:
            use_channels: 输出几个channel。
            mask_file_lst: list, 预测结果保存的文件名列表。
            tta: bool, 预测时是否进行数据增强。
            is_save_npy: bool, 是否保存npy文件。
            is_save_mask0: bool
            is_save_mask1: bool
            result_save_dir: str, 结果保存的目录路径。
        Returns: 4-d numpy array, predicted result.

        """
        imgs = self.read_images(image_path_lst)
        imgs = DataSet.preprocess(imgs, mode="image")
        if tta:
            pred = tta_predict(self.model, imgs, batch_size=batch_size)
        else:
            pred = self.predict_old(imgs, batch_size=batch_size, use_channels=use_channels)
        if mask_file_lst is None:
            mask_file_lst = [os.path.basename(x) for x in image_path_lst]
        if is_save_npy:
            # 保存npy文件
            npy_dir = os.path.join(result_save_dir, "npy")
            self.save_npy(pred, mask_file_lst, npy_dir)
        if is_save_mask0:
            mask_nb = 0
            mask_save_dir = os.path.join(result_save_dir, "mask{}".format(mask_nb))
            self.save_mask(pred, mask_file_lst, mask_nb=mask_nb, result_save_dir=mask_save_dir)
        if is_save_mask1:
            mask_nb = 1
            mask_save_dir = os.path.join(result_save_dir, "mask{}".format(mask_nb))
            self.save_mask(pred, mask_file_lst, mask_nb=mask_nb, result_save_dir=mask_save_dir)
        return pred

    @staticmethod
    def read_npy(npy_root_dir, mask_file_lst):
        npy_file_path_lst = [os.path.join(npy_root_dir, x) for x in mask_file_lst]
        pred = [np.load(x) for x in npy_file_path_lst]
        pred = np.array(pred)
        return pred

    @staticmethod
    def save_npy(pred, mask_file_lst, result_save_dir):
        if not os.path.isdir(result_save_dir):
            os.makedirs(result_save_dir)
        pred_np_path_lst = [os.path.join(result_save_dir, os.path.splitext(x)[0] + ".npy") for x in
                            mask_file_lst]
        for i in range(len(pred)):
            np.save(pred_np_path_lst[i], pred[i])

    @staticmethod
    def save_mask(pred, mask_file_lst, mask_nb, result_save_dir):
        """

        Args:
            pred: 4-d numpy array, (b, h, w, c)
            mask_file_lst:
            mask_nb:
            result_save_dir:

        Returns:

        """
        if not os.path.isdir(result_save_dir):
            os.makedirs(result_save_dir)
        masks = pred[..., mask_nb]
        mask_file_path_lst = [os.path.join(result_save_dir, x) for x in mask_file_lst]
        # 将预测结果转换为0-1数组。
        masks = ModelDeployment.postprocess(masks)
        # 将0-1数组转换为0-255数组。
        masks = DataSet.de_preprocess(masks, mode="mask")
        for i in range(len(pred)):
            cv2.imwrite(mask_file_path_lst[i], masks[i])

    # def predict_from_files(self, image_path_lst, batch_size=5, use_channels=1, result_save_dir=None,
    #                        mask_file_lst=None, use_npy=False):
    #     """
    #     给定图片路径列表，返回预测结果（未处理过的），如果指定了预测结果保存路径，则保存预测结果（已处理过的）。
    #     如果指定了预测结果保存的文件名列表，则该列表必须与image_path_lst一致。
    #     Args:
    #         image_path_lst: list.
    #         batch_size:
    #         use_channels: 输出几个channel。
    #         result_save_dir:　预测结果保存的目录。
    #         mask_file_lst: list, 预测结果保存的文件名列表。
    #
    #     Returns: predicted result.
    #
    #     """
    #     imgs = self.read_images(image_path_lst)
    #     imgs = DataSet.preprocess(imgs, mode="image")
    #     pred = self.predict(imgs, batch_size=batch_size, use_channels=use_channels)
    #     if result_save_dir:
    #         if not os.path.isdir(result_save_dir):
    #             os.makedirs(result_save_dir)
    #         if use_channels == 2:
    #             if use_npy:
    #                 # 保存npy文件
    #                 pred_np_path_lst = [os.path.join(result_save_dir, os.path.splitext(x)[0] + ".npy") for x in
    #                                     mask_file_lst]
    #                 for i in range(len(pred)):
    #                     np.save(pred_np_path_lst[i], pred[i])
    #             else:
    #                 mask0_save_dir = os.path.join(result_save_dir, "mask0")
    #                 mask1_save_dir = os.path.join(result_save_dir, "mask1")
    #                 if not os.path.isdir(mask0_save_dir):
    #                     os.makedirs(mask0_save_dir)
    #                 if not os.path.isdir(mask1_save_dir):
    #                     os.makedirs(mask1_save_dir)
    #                 pred_mask0 = pred[..., 0]
    #                 pred_mask1 = pred[..., 1]
    #
    #                 mask0_path_lst = [os.path.join(mask0_save_dir, x) for x in mask_file_lst]
    #                 mask1_path_lst = [os.path.join(mask1_save_dir, x) for x in mask_file_lst]
    #                 # 将预测结果转换为0-1数组。
    #                 pred_mask0 = self.postprocess(pred_mask0)
    #                 pred_mask1 = self.postprocess(pred_mask1)
    #                 # 将0-1数组转换为0-255数组。
    #                 pred_mask0 = DataSet.de_preprocess(pred_mask0, mode="mask")
    #                 pred_mask1 = DataSet.de_preprocess(pred_mask1, mode="mask")
    #                 for i in range(len(pred)):
    #                     cv2.imwrite(mask0_path_lst[i], pred_mask0[i])
    #                     cv2.imwrite(mask1_path_lst[i], pred_mask1[i])
    #
    #         else:
    #             if use_npy:
    #                 raise NotImplemented
    #             else:
    #                 mask_path_lst = [os.path.join(result_save_dir, x) for x in mask_file_lst]
    #                 # 将预测结果转换为0-1数组。
    #                 pred = self.postprocess(pred)
    #                 # 将0-1数组转换为0-255数组。
    #                 pred = DataSet.de_preprocess(pred, mode="mask")
    #                 for i in range(len(pred)):
    #                     cv2.imwrite(mask_path_lst[i], pred[i])
    #         print("预测结果已保存。")
    #     if use_channels == 2:
    #         if use_npy:
    #             pass
    #         else:
    #             pred_mask0 = pred[..., 0]
    #             pred_mask1 = pred[..., 1]
    #             pred_mask1 = np.where(pred_mask1 > 0.5,
    #                                   1,
    #                                   0)
    #             pred = np.multiply(pred_mask0, pred_mask1)
    #     return pred

    def predict_and_show(self, image, show_output_channels):
        """
        
        Args:
            img: str(image path) or numpy array(b=1, h=576, w=576, c=1)
            show_output_channels: 1 or 2

        Returns:

        """
        if isinstance(image, str):
            images_src = self.read_images([image])
        else:
            images_src = image
        img = DataSet.preprocess(images_src, mode="image")
        predict_mask = self.predict(img, 1, use_channels=show_output_channels)
        predict_mask = np.squeeze(predict_mask, axis=0)
        predict_mask = self.postprocess(predict_mask)
        predict_mask = DataSet.de_preprocess(predict_mask, mode="mask")
        if show_output_channels == 2:
            mask0 = predict_mask[..., 0]
            mask1 = predict_mask[..., 1]
            image_c3 = np.concatenate([np.squeeze(images_src, axis=0) for i in range(3)], axis=-1)
            image_mask0 = apply_mask(image_c3, mask0, color=[255, 106, 106], alpha=0.5)
            # result = np.concatenate((np.squeeze(images_src, axis=[0, -1]), mask0, mask1, image_mask0), axis=1)
            plt.imshow(image_mask0)
        else:
            result = np.concatenate((np.squeeze(images_src, axis=[0, -1]), predict_mask), axis=1)
            plt.imshow(result, cmap="gray")

        plt.show()

    def predict_from_h5data(self, h5_data_path, val_fold_nb, use_channels, is_train=False, save_dir=None,
                            random_k_fold=False, random_k_fold_npy=None, input_channels=1,
                            output_channels=2, random_crop_size=None, mask_nb=0, batch_size=4
                            ):
        dataset = DataSet(h5_data_path, val_fold_nb, random_k_fold=random_k_fold, random_k_fold_npy=random_k_fold_npy,
                          input_channels=input_channels,
                          output_channels=output_channels, random_crop_size=random_crop_size, mask_nb=mask_nb,
                          batch_size=batch_size)
        images, _ = dataset.prepare_data(is_train)
        pred = self.predict(images, batch_size, use_channels=use_channels)

        if save_dir:
            keys = dataset.get_keys(is_train)
            mask_file_lst = ["{:03}.tif".format(int(key)) for key in keys]
            self.save_mask(pred, mask_file_lst, mask_nb, result_save_dir=save_dir)
        return pred

    def predict_from_h5data_old(self, h5_data_path, val_fold_nb, is_train=False, save_dir=None,
                                color_lst=None):
        dataset = DataSet(h5_data_path, val_fold_nb)

        images = dataset.get_images(is_train=is_train)
        imgs_src = np.concatenate([images for i in range(3)], axis=-1)
        masks = dataset.get_masks(is_train=is_train, mask_nb=0)
        masks = np.squeeze(masks, axis=-1)
        print("predicting ...")
        y_pred = self.predict(dataset.preprocess(images, mode="image"), batch_size=4, use_channels=1)
        y_pred = self.postprocess(y_pred)
        y_pred = DataSet.de_preprocess(y_pred, mode="mask")
        print(y_pred.shape)

        if save_dir:
            keys = dataset.get_keys(is_train)
            if color_lst is None:
                color_gt = [255, 106, 106]
                color_pred = [0, 191, 255]
                # color_pred = [255, 255, 0]
            else:
                color_gt = color_lst[0]
                color_pred = color_lst[1]
            # BGR to RGB
            imgs_src = imgs_src[..., ::-1]
            image_masks = [apply_mask(image, mask, color_gt, alpha=0.5) for image, mask in zip(imgs_src, masks)]
            image_preds = [apply_mask(image, mask, color_pred, alpha=0.5) for image, mask in zip(imgs_src, y_pred)]
            dst_image_path_lst = [os.path.join(save_dir, "{:03}.tif".format(int(key))) for key in keys]
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            image_mask_preds = np.concatenate([imgs_src, image_masks, image_preds], axis=2)
            for i in range(len(image_masks)):
                cv2.imwrite(dst_image_path_lst[i], image_mask_preds[i])
            print("Done.")
        else:
            return y_pred

    def predict_and_save_stage1_masks(self, h5_data_path, h5_result_saved_path, fold_k=0, batch_size=4):
        """
        从h5data中读取images进行预测，并把预测mask保存进h5data中。
        Args:
            h5_data_path: str, 存放有训练数据的h5文件路径。
            batch_size: int, 批大小。

        Returns: None.

        """

        f_result = h5py.File(h5_result_saved_path, "a")
        try:
            stage1_predict_masks_grp = f_result.create_group("stage1_fold{}_predict_masks".format(fold_k))
        except:
            stage1_predict_masks_grp = f_result["stage1_fold{}_predict_masks".format(fold_k)]

        dataset = DataSet(h5_data_path, fold_k)

        images_train = dataset.get_images(is_train=True)
        images_val = dataset.get_images(is_train=False)
        keys_train = dataset.get_keys(is_train=True)
        keys_val = dataset.get_keys(is_train=False)
        images = np.concatenate([images_train, images_val], axis=0)
        keys = np.concatenate([keys_train, keys_val], axis=0)
        print("predicting ...")
        images = dataset.preprocess(images, mode="image")
        y_pred = self.predict(images, batch_size, use_channels=1)
        print(y_pred.shape)
        print("Saving predicted masks ...")
        for i, key in enumerate(keys):
            stage1_predict_masks_grp.create_dataset(key, dtype=np.float32, data=y_pred[i])
        print("Done.")

    # def predict_and_save_image_masks(self, image_path_lst, save_dir, batch_size=32, color=None, alpha=0.5):
    #     images = self.read_images(image_path_lst)
    #     images = DataSet.preprocess(images, mode="image")
    #     masks = self.predict(images, batch_size, use_channels=1)
    #     masks = np.where(masks > 0.5, 255, 0)
    #
    #     images = [np.concatenate([image for i in range(3)], axis=-1) for image in images]
    #     masks = [mask[..., 0] for mask in masks]
    #
    #     if color is None:
    #         color = [0, 191, 255]
    #     image_masks = [apply_mask(image, mask, color, alpha) for image, mask in zip(images, masks)]
    #     dst_image_path_lst = [os.path.join(save_dir, os.path.basename(x)) for x in image_path_lst]
    #     if not os.path.isdir(save_dir):
    #         os.makedirs(save_dir)
    #     for i in range(len(image_masks)):
    #         scipy.misc.toimage(image_masks[i], cmin=0.0, cmax=...).save(dst_image_path_lst[i])


def get_acc(model_def, model_weights, h5_data_path, val_fold_nb, is_train=False):
    dataset = DataSet(h5_data_path, val_fold_nb)
    images, masks = dataset.prepare_1i_1o_data(is_train=is_train, mask_nb=0)

    model_obj = ModelDeployment(model_def, model_weights)

    y_pred = model_obj.predict(images, batch_size=4)

    K.clear_session()
    if y_pred.shape[-1] == 2:
        y_pred = y_pred[..., 0]
    print(y_pred.shape)

    y_true = masks
    acc = get_pixel_wise_acc(y_true, y_pred)
    with tf.Session() as sess:
        print(sess.run(acc))


def inference_2stages_from_files(model_def_stage1, model_weights_stage1, model_def_stage2, model_weights_stage2,
                                 file_dir, pred_save_dir):
    if not os.path.isdir(pred_save_dir):
        os.makedirs(pred_save_dir)
    model_obj = ModelDeployment(model_def_stage1, model_weights_stage1)
    file_path_lst = get_file_path_list(file_dir, ext=".tif")
    dst_file_path_lst = [os.path.join(pred_save_dir, os.path.basename(x)) for x in file_path_lst]

    imgs_src = model_obj.read_images(file_path_lst)
    imgs = DataSet.preprocess(imgs_src, mode="image")
    pred_stage1 = model_obj.predict(imgs, batch_size=5, use_channels=1)
    pred_stage1 = np.expand_dims(pred_stage1, axis=-1)
    input_stage2 = np.concatenate([imgs_src, pred_stage1], axis=-1)
    del model_obj
    print(pred_stage1.shape)
    print(input_stage2.shape)
    model_obj = ModelDeployment(model_def_stage2, model_weights_stage2)

    pred = model_obj.predict(input_stage2, batch_size=5, use_channels=1)
    pred = model_obj.postprocess(pred)
    pred = DataSet.de_preprocess(pred, mode="mask")
    for i in range(len(pred)):
        cv2.imwrite(dst_file_path_lst[i], pred[i])


#
# def inference_2stages(model_def_stage1, model_weights_stage1, model_def_stage2, model_weights_stage2, h5_data_path,
#                       val_fold_nb, pred_save_dir, image_masks_save_dir):
#     dataset = DataSet(h5_data_path, val_fold_nb)
#     # keys = dataset.val_keys
#     keys = dataset.train_keys
#     images = get_images(dataset.f_h5, keys)
#     if len(get_file_path_list(pred_save_dir, ".npy")) == 0:
#         model = model_def_stage1
#         print("Loading stage1 weights ...")
#         model.load_weights(model_weights_stage1)
#         # model.summary()
#
#         stage1_outputs = model.predict(images, 4)
#         print(stage1_outputs.shape)
#         # K.clear_session()
#
#         stage2_inputs = np.concatenate([images, stage1_outputs], axis=-1)
#         model = model_def_stage2
#         print("Loading stage2 weights ...")
#         model.load_weights(model_weights_stage2)
#         outputs = model.predict(stage2_inputs, 4)
#         predicted_masks = np.squeeze(outputs, axis=-1)
#
#         if not os.path.isdir(pred_save_dir):
#             os.makedirs(pred_save_dir)
#         np_save_path = os.path.join(pred_save_dir, "predicted_masks_2stages_fold{}".format(val_fold_nb) + ".npy")
#         np.save(np_save_path, predicted_masks)
#     else:
#         np_save_path = os.path.join(pred_save_dir, "predicted_masks_2stages_fold{}".format(val_fold_nb) + ".npy")
#
#         predicted_masks = np.load(np_save_path)
#
#     masks = np.where(predicted_masks > 0.5, 1, 0)
#     images = de_preprocess(images)
#     images = [np.concatenate([image for i in range(3)], axis=-1) for image in images]
#
#     color = [0, 191, 255]
#     image_masks = [apply_mask(image, mask, color, alpha=0.5) for image, mask in zip(images, masks)]
#     dst_image_path_lst = [os.path.join(image_masks_save_dir, "{:03}.tif".format(int(x))) for x in keys]
#     if not os.path.isdir(image_masks_save_dir):
#         os.makedirs(image_masks_save_dir)
#     for i in range(len(image_masks)):
#         cv2.imwrite(dst_image_path_lst[i], image_masks[i])


def __main():
    # Set test mode
    K.set_learning_phase(0)
    np.set_printoptions(threshold=np.inf)
    # do_infer_2stages()
    # do_get_acc()
    # do_evaluate()
    # infer_do()
    # inference_and_sub()
    # make_sub()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
