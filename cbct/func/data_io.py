#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-7-17
# Task: 
# Insights:
import json
from datetime import datetime
import os
from time import sleep

import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


class DataSet(object):
    def __init__(self, h5_data_path, val_fold_nb, random_k_fold=False, random_k_fold_npy=None, input_channels=1,
                 output_channels=2, random_crop_size=(256, 256), mask_nb=0, batch_size=4

                 ):
        f_h5 = h5py.File(h5_data_path, 'r')
        k_fold_map = json.loads(f_h5["k_fold_map"].value)
        folds = k_fold_map.keys()
        if random_k_fold:
            val_fold_nb = int(val_fold_nb)
            if os.path.isfile(random_k_fold_npy):
                all_keys_array = np.load(random_k_fold_npy)
            else:
                all_keys = list(f_h5["images"].keys())
                np.random.shuffle(all_keys)
                all_keys_array = np.array(all_keys).reshape(10, -1)
                np.save(random_k_fold_npy, all_keys_array)
            val_keys = all_keys_array[val_fold_nb]
            train_keys = np.setdiff1d(all_keys_array.flatten(), val_keys)
        else:
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

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.random_crop_size = random_crop_size
        self.mask_nb = mask_nb
        self.batch_size = batch_size

    def get_train_val_steps(self, is_train):
        if is_train:
            return len(self.train_keys) // self.batch_size
        else:
            return len(self.val_keys) // self.batch_size

    def prepare_data(self, is_train):
        input_channels = self.input_channels
        output_channels = self.output_channels
        mask_nb = self.mask_nb
        if input_channels == 1 and output_channels == 2:
            return self._prepare_1i_2o_data(is_train)
        elif input_channels == 1 and output_channels == 1:
            return self._prepare_1i_1o_data(is_train, mask_nb)
        elif input_channels == 3 and output_channels == 2:
            return self._prepare_3i_2o_data(is_train)
        elif input_channels == 3 and output_channels == 1:
            return self._prepare_3i_1o_data(is_train, mask_nb)

    def prepare_stage1_data(self, is_train):
        images = self.get_images(is_train)

        masks = self.get_masks(is_train, 1)

        return images, masks

    def prepare_stage2_data(self, is_train):
        images = self.get_images(is_train)

        stage1_predict_masks = self.get_stage1_predict_masks(is_train)
        images = np.concatenate([images, stage1_predict_masks], axis=-1)
        masks = self.get_masks(is_train, 0)

        return images, masks

    def _prepare_3i_2o_data(self, is_train):
        images = self.get_images(is_train)
        images = np.concatenate([images for i in range(3)], axis=-1)

        masks_0 = self.get_masks(is_train, 0)
        masks_1 = self.get_masks(is_train, 1)
        masks = np.concatenate([masks_0, masks_1], axis=-1)
        return images, masks

    def _prepare_3i_1o_data(self, is_train, mask_nb=0):
        images = self.get_images(is_train)
        images = np.concatenate([images for i in range(3)], axis=-1)

        masks = self.get_masks(is_train, mask_nb)

        return images, masks

    def _prepare_1i_2o_data(self, is_train):
        images = self.get_images(is_train)

        masks_0 = self.get_masks(is_train, 0)
        masks_1 = self.get_masks(is_train, 1)
        masks = np.concatenate([masks_0, masks_1], axis=-1)
        return images, masks

    def _prepare_1i_1o_data(self, is_train, mask_nb=0):
        images = self.get_images(is_train)

        masks = self.get_masks(is_train, mask_nb)

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
            imgs = np.where(images > 127.5, 1, 0)
        return imgs

    @staticmethod
    def random_crop(img, random_crop_size, seed=0):
        """
        Random crop image.
        Args:
            img: 3-d numpy array, such as (h, w, c) and h == w.
            random_crop_size: tuple, such as (random_crop_height, random_crop_width).

        Returns:

        """
        np.random.seed(seed)
        assert len(img.shape) == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y + dy), x:(x + dx), :]

    def preprocess_x(self, image):
        return self.preprocess(image, mode="image")

    def preprocess_y(self, image):
        return self.preprocess(image, mode="mask")

    def get_custom_data_generator(self, is_train, keras_data_gen_param, seed=1248):
        epoch_idx = 0
        steps = self.get_train_val_steps(is_train)
        images, masks = self.prepare_data(is_train)
        print("is_train:", is_train)
        print("src input shape: ({}, {})".format(images.shape, masks.shape))
        images_cropped = np.zeros(
            shape=(images.shape[0], self.random_crop_size[0], self.random_crop_size[1], images.shape[-1]))
        masks_cropped = np.zeros(
            shape=(images.shape[0], self.random_crop_size[0], self.random_crop_size[1], masks.shape[-1]))
        # initialization
        for i in range(len(images)):
            seed = seed + epoch_idx + i
            images_cropped[i] = self.random_crop(images[i], self.random_crop_size, seed)
            masks_cropped[i] = self.random_crop(masks[i], self.random_crop_size, seed)
        print("model input shape: ({}, {})".format(images_cropped.shape, masks_cropped.shape))


        keras_data_gen_x = ImageDataGenerator(**keras_data_gen_param, preprocessing_function=self.preprocess_x)
        keras_data_gen_y = ImageDataGenerator(**keras_data_gen_param, preprocessing_function=self.preprocess_y)

        x_generator = keras_data_gen_x.flow(images_cropped, None, batch_size=self.batch_size, shuffle=True, seed=seed)
        y_generator = keras_data_gen_y.flow(masks_cropped, None, batch_size=self.batch_size, shuffle=True, seed=seed)
        keras_data_generator = zip(x_generator, y_generator)

        step = 0
        while 1:
            if step == steps - 1:
                np.random.seed(seed)
                np.random.shuffle(images)
                np.random.seed(seed)
                np.random.shuffle(masks)
                for i in range(len(images)):
                    seed = seed + epoch_idx + i
                    images_cropped[i] = self.random_crop(images[i], self.random_crop_size, seed)
                    masks_cropped[i] = self.random_crop(masks[i], self.random_crop_size, seed)

                x_generator = keras_data_gen_x.flow(images_cropped, None, batch_size=self.batch_size, shuffle=True,
                                                    seed=seed)
                y_generator = keras_data_gen_y.flow(masks_cropped, None, batch_size=self.batch_size, shuffle=True,
                                                    seed=seed)
                keras_data_generator = zip(x_generator, y_generator)
            for step in range(steps):
                x, y = next(keras_data_generator)
                yield x, y
            epoch_idx += 1

    def get_keras_data_generator(self, is_train, keras_data_gen_param, seed=1248):
        images, masks = self.prepare_data(is_train)
        print("is_train:", is_train)
        print("model input shape: ({}, {})".format(images.shape, masks.shape))
        keras_data_gen_x = ImageDataGenerator(**keras_data_gen_param, preprocessing_function=self.preprocess_x)
        keras_data_gen_y = ImageDataGenerator(**keras_data_gen_param, preprocessing_function=self.preprocess_y)

        x_generator = keras_data_gen_x.flow(images, None, self.batch_size, shuffle=True, seed=seed)
        y_generator = keras_data_gen_y.flow(masks, None, self.batch_size, shuffle=True, seed=seed)
        keras_data_generator = zip(x_generator, y_generator)
        return keras_data_generator


def seed_test():
    np.random.seed(12)
    l1 = [np.random.randint(100) for i in range(10)]
    print(l1)
    np.random.seed(12)
    l2 = [np.random.randint(100) for i in range(10)]
    print(l2)


def __main():
    # ori = np.arange(16).reshape(4, 4, 1)
    # s1 = DataSet.random_crop(ori, (2, 2), 3)
    # s2 = DataSet.random_crop(ori, (2, 2), 4)
    # print(s1)
    # print("*" * 20)
    # print(s2)
    seed_test()
    # h5_data_path = ""
    pass


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
