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
from func.data_io import DataSet
from func.model_inception_resnet_v2_gn import get_inception_resnet_v2_unet_sigmoid_gn
from keras.models import load_model
from matplotlib import pyplot as plt
from tqdm import tqdm
import keras.backend as K
import tensorflow as tf
from module.competition_utils import get_pixel_wise_acc
from module.utils_public import apply_mask, get_file_path_list


class ModelDeployment(object):
    def __init__(self, model_def, model_weights):
        model = model_def

        print("Loading weights ...")
        model.load_weights(model_weights)
        model.summary()
        self.model = model

    def evaluate(self, x_val, y_val):
        # Score trained model.
        scores = self.model.evaluate(x_val, y_val, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    @staticmethod
    def postprocess(masks):
        return DataSet.preprocess(masks, mode="mask")

    def predict_and_save_stage1_masks(self, h5_data_path, h5_result_saved_path, fold_k=0, batch_size=32):
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

        images, masks = dataset.prepare_1i_1o_data(is_train=is_train, mask_nb=0)
        print("predicting ...")
        y_preds = self.model.predict(images, batch_size=4)
        print(y_preds.shape)

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

    def read_images(self, image_path_lst):
        print("Loading images ...")
        imgs = []
        for image_path in tqdm(image_path_lst):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, -1)
            imgs.append(img)
        return np.array(imgs)

    def predict(self, images, batch_size, use_channels=1):
        outputs = self.model.predict(images, batch_size)
        if use_channels == 1:
            outputs = outputs[..., 0]
        return outputs

    def predict_from_h5data(self, h5_data_path, val_fold_nb, is_train=False, save_dir=None,
                            color_lst=None):
        dataset = DataSet(h5_data_path, val_fold_nb)

        images = dataset.get_images(is_train=is_train)
        imgs_src = np.concatenate([images for i in range(3)], axis=-1)
        masks = dataset.get_masks(is_train=is_train, mask_nb=0)
        masks = np.squeeze(masks, axis=-1)
        print("predicting ...")
        y_preds = self.model.predict(dataset.preprocess(images, mode="image"), batch_size=4)
        y_preds = self.postprocess(y_preds)
        y_preds = DataSet.de_preprocess(y_preds, mode="mask")
        y_preds = y_preds[..., 0]
        print(y_preds.shape)

        if save_dir:
            keys = dataset._get_keys(is_train)
            if color_lst is None:
                color_gt = [255, 106, 106]
                color_pred = [0, 191, 255]
                # color_pred = [255, 255, 0]
            else:
                color_gt = color_lst[0]
                color_pred = color_lst[1]

            image_masks = [apply_mask(image, mask, color_gt, alpha=0.5) for image, mask in zip(imgs_src, masks)]
            image_preds = [apply_mask(image, mask, color_pred, alpha=0.5) for image, mask in zip(imgs_src, y_preds)]
            dst_image_path_lst = [os.path.join(save_dir, "{:03}.tif".format(int(key))) for key in keys]
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            image_mask_preds = np.concatenate([imgs_src, image_masks, image_preds], axis=2)
            for i in range(len(image_masks)):
                cv2.imwrite(dst_image_path_lst[i], image_mask_preds[i])
            print("Done.")
        else:
            return y_preds

    # def predict_and_save_image_masks(self, image_path_lst, save_dir, batch_size=32, color=None, alpha=0.5):
    #     images = self.read_images(image_path_lst)
    #     images = DataSet.preprocess(images, mode="image")
    #     masks = self.predict(images, batch_size)
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


def do_get_acc():
    h5_data_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data_0717.hdf5"
    val_fold_nb = 1
    output_channels = 2
    is_train = False
    model_def = get_inception_resnet_v2_unet_sigmoid(weights=None, output_channels=output_channels)
    model_weights = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/final_inception_resnet_v2_bn_fold1_1i_2o.h5"
    get_acc(model_def, model_weights, h5_data_path, val_fold_nb, is_train)


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


# def do_infer_2stages():
#     model_def_stage1 = get_inception_resnet_v2_unet_sigmoid(input_shape=(576, 576, 1), weights=None, output_channels=1)
#     model_weights_stage1 = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/inception_resnet_v2_stage1_fold1.h5"
#     model_def_stage2 = get_inception_resnet_v2_unet_sigmoid(input_shape=(576, 576, 2), weights=None, output_channels=1)
#     model_weights_stage2 = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/inception_resnet_v2_stage2_fold1.h5"
#     h5_data_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data_0717.hdf5"
#     val_fold_nb = 1
#     pred_save_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/pred_masks_2stage_fold1_train_0721"
#     image_masks_save_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/pred_image_masks_2stage_fold1_train_0721"
#     inference_2stages(model_def_stage1, model_weights_stage1, model_def_stage2, model_weights_stage2, h5_data_path,
#                       val_fold_nb, pred_save_dir, image_masks_save_dir)


def infer_do():
    h5_data_path = "/home/zj/helloworld/study/njai_challenge/cbct/inputs/data_0717.hdf5"
    result_save_dir = "/media/zj/share/data/njai_2018/cbct/result/image_mask_preds20180726"
    model_weights = "/home/zj/helloworld/study/njai_challenge/cbct/model_weights/final_inception_resnet_v2_gn_fold1_1i_2o.h5"
    model_def = get_inception_resnet_v2_unet_sigmoid_gn(weights=None, output_channels=2)
    model_obj = ModelDeployment(model_def, model_weights)
    # model_obj.predict_from_h5data(h5_data_path, val_fold_nb=1, is_train=False, save_dir=result_save_dir)
    # model_obj.predict_from_h5data(h5_data_path, val_fold_nb=1, is_train=True, save_dir=result_save_dir)


    dataset = DataSet(h5_data_path, val_fold_nb=1)
    keys = dataset.val_keys
    images = dataset.get_image_by_key(keys[8])
    images = np.expand_dims(images, axis=0)
    images = np.expand_dims(images, axis=-1)
    model_obj.predict_and_show(images, show_output_channels=2)


def __main():
    # Set test mode
    K.set_learning_phase(0)
    np.set_printoptions(threshold=np.inf)
    # do_infer_2stages()
    # do_get_acc()
    infer_do()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
