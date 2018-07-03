#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-20
# Task: 
# Insights:
import glob
import os
import subprocess
from datetime import datetime

import h5py
import numpy as np
import scipy.misc
from PIL import Image

from module.utils_public import apply_mask


def save_labelme2_mask():
    """
    把labelme工具制作的标注数据转换为mask.
    Returns:

    """
    # .json文件所在的目录。
    labelme_mask_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/mask_image_1"
    # labelme_json_to_dataset转换的结果保存目录。
    json_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/gum_json"
    # 新生成的mask保存的目录。
    new_mask_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/gum_mask"
    if not os.path.isdir(json_dir):
        os.makedirs(json_dir)
    if not os.path.isdir(new_mask_dir):
        os.makedirs(new_mask_dir)

    json_lst = glob.glob(os.path.join(labelme_mask_dir, "*.json"))
    for jsonfile in json_lst:
        basename = os.path.splitext(os.path.basename(jsonfile))[0]
        dst_dir = os.path.join(json_dir, basename + "_json")
        if not os.path.isdir(dst_dir):
            shell = "/home/topsky/anaconda3/bin/labelme_json_to_dataset {} -o {}".format(jsonfile, dst_dir)
            subprocess.run(shell, shell=True)
        label_png = np.array(Image.open(os.path.join(dst_dir, "label.png")))
        scipy.misc.imsave(os.path.join(new_mask_dir, basename + ".tif"), label_png)


def save_mask_image():
    """
    合并image与mask并保存。
    Returns:

    """
    data_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct"
    images_dir = os.path.join(data_dir, "train")
    masks_dir = os.path.join(data_dir, "label")
    # 新合并图的保存路径。
    mask_image_dir = os.path.join(data_dir, "mask_image_1")
    if not os.path.isdir(mask_image_dir):
        os.makedirs(mask_image_dir)
    basename_lst = next(os.walk(images_dir))[2]
    color = [0, 191, 255]
    alpha = 0.5
    for basename in basename_lst:
        image = Image.open(os.path.join(images_dir, basename))
        image = np.array(image)
        if len(image.shape) == 2:
            image = np.concatenate([np.expand_dims(image, axis=-1) for i in range(3)], axis=-1)
        mask = Image.open(os.path.join(masks_dir, basename))
        mask = np.array(mask)
        if len(mask.shape) == 3:
            assert mask[:, :, 0].all() == mask[:, :, 1].all() and mask[:, :, 0].all() == mask[:, :, 2].all()
            mask = mask[:, :, 0]
        mask_image = apply_mask(image, mask, color, alpha=alpha)
        # mask_image是否与image一起显示。
        # mask_image = np.concatenate([image, mask_image], axis=1)
        scipy.misc.toimage(mask_image, cmin=0.0, cmax=...).save(os.path.join(mask_image_dir, basename))


def make_hdf5_database():
    """
    将image与mask保存在hdf5数据库里。
    Returns:

    """
    data_root = "/media/topsky/HHH/jzhang_root/data/njai/cbct"
    hdf5_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data.hdf5"
    f = h5py.File(hdf5_path, "w")
    grp_x = f.create_group("images")
    grp_y_1 = f.create_group("mask_1")
    grp_y_2 = f.create_group("mask_2")
    file_basename_lst = next(os.walk(os.path.join(data_root, "train")))[2]
    image_file_path_lst = [os.path.join(data_root, "train", x) for x in file_basename_lst]
    mask_1_file_path_lst = [os.path.join(data_root, "label", x) for x in file_basename_lst]
    mask_2_file_path_lst = [os.path.join(data_root, "gum_mask", x) for x in file_basename_lst]

    for i in range(len(file_basename_lst)):
        idx = '{:08}'.format(i + 1)
        # 保存image,原样保存。
        image = Image.open(image_file_path_lst[i])
        image = np.array(image, dtype=np.uint8)
        # if len(image.shape) == 3:
        #     image = image[:, :, 0]
        grp_x.create_dataset(idx, dtype=np.uint8, data=image)  # [0, 255]
        # 保存mask_tooth_root,如果通道数不是１,则只保存通道１的图片。
        mask_1 = Image.open(mask_1_file_path_lst[i])
        mask_1 = np.array(mask_1, dtype=np.uint8)
        if len(mask_1.shape) == 3:
            mask_1 = mask_1[:, :, 0]
        mask_1 = np.expand_dims(mask_1, -1)
        grp_y_1.create_dataset(idx, dtype=np.uint8, data=mask_1)
        # 保存mask_gum,如果通道数不是１,则只保存通道１的图片。
        mask_2 = Image.open(mask_2_file_path_lst[i])
        mask_2 = np.array(mask_2, dtype=np.uint8)
        if len(mask_2.shape) == 3:
            mask_2 = mask_2[:, :, 0]
        mask_2 = np.expand_dims(mask_2, -1)
        grp_y_2.create_dataset(idx, dtype=np.uint8, data=mask_2)
        if i % 10 == 0:
            print("Saving images and labels ... ", i)


def add_train_val_id_hdf5():
    hdf5_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data.hdf5"
    f = h5py.File(hdf5_path, mode="r+")
    # del f["train_id"], f["val_id"]
    # exit()
    keys = list(f["images"].keys())
    print(keys)
    print(type(keys[0]))
    keys = [x.encode() for x in keys]
    print(keys)
    print(type(keys[0]))
    keys = [np.void(x) for x in keys]
    print(keys)
    print(type(keys[0]))
    np.random.shuffle(keys)
    print(keys)
    split = int(len(keys) * 0.9)
    for name in f:
        print(name)
    f.create_dataset("train_id", data=keys[:split])
    f.create_dataset("val_id", data=keys[split:])
    v = f["train_id"].value
    v = [x.tostring().decode() for x in v]
    print(v)
    print(len(v))
    f.close()


def __main():
    # make_hdf5_database()
    # add_train_val_id_hdf5()
    # save_mask_image()
    # show_image()
    # save_labelme2_mask()
    pass


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
