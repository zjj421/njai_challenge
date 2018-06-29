#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-20
# Task: 
# Insights:
import glob
import os
import subprocess
from datetime import datetime

import cv2
import h5py
import numpy as np
from PIL import Image

from cbct.func.utils import DataGeneratorCustom
from module.utils_public import apply_mask
from matplotlib import pyplot as plt
import scipy.misc


def show_image():
    np.set_printoptions(threshold=np.nan)
    image_path = "/media/zj/share/data/njai_2018/cbct/train/001.tif"
    mask_path = "/media/zj/share/data/njai_2018/cbct/label/001.tif"
    # color = np.random.rand(3)
    color = [0, 191, 255]
    print(color)
    image = Image.open(image_path)
    image = np.array(image)
    mask = Image.open(mask_path)
    mask = np.array(mask)
    mask = mask[:, :, 0]
    print(image.shape)
    print(mask.shape)
    mask_image = apply_mask(image, mask, color, alpha=0.5)
    image = np.concatenate([image, mask_image], axis=1)
    plt.imshow(image)
    plt.show()
    # scipy.misc.toimage(image, cmin=0.0, cmax=...).save('outfile.jpg')


def save_labelme2_mask():
    labelme_mask_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/mask_image_1"
    json_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/gum_json"
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
    data_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct"
    images_dir = os.path.join(data_dir, "train")
    masks_dir = os.path.join(data_dir, "label")
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
        # image2 = np.concatenate([image, mask_image], axis=1)
        scipy.misc.toimage(mask_image, cmin=0.0, cmax=...).save(os.path.join(mask_image_dir, basename))


def data_analysis():
    np.set_printoptions(threshold=np.nan)
    data_root = "/media/zj/share/data/njai_2018/cbct"
    file_basename_lst = next(os.walk(os.path.join(data_root, "train")))[2]
    train_file_path_lst = [os.path.join(data_root, "train", x) for x in file_basename_lst]
    label_file_path_lst = [os.path.join(data_root, "label", x) for x in file_basename_lst]
    train_shape = []
    label_shape = []
    for i, file_path in enumerate(train_file_path_lst):
        im = Image.open(file_path)
        im = np.array(im, dtype=np.uint8)
        if len(im.shape) == 3:
            print(file_path)
            assert im[:, :, 0].all() == im[:, :, 1].all() and im[:, :, 0].all() == im[:, :, 2].all()
        train_shape.append(im.shape)
    for i, file_path in enumerate(label_file_path_lst):
        im = Image.open(file_path)
        im = np.array(im, dtype=np.uint8)
        label_shape.append(im.shape)
    print(set(train_shape))
    print(set(label_shape))
    # for i, s in enumerate(train_shape):
    #     if s == (576, 576, 3):
    #         print(train_file_path_lst[i])


# deprecated
# def make_database():
#     data_root = "/media/topsky/HHH/jzhang_root/data/njai/cbct"
#     lmdb_x_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/train_lmdb_x"
#     lmdb_y_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/train_lmdb_y"
#     lmdb_x = LMDB(lmdb_x_path)
#     lmdb_y = LMDB(lmdb_y_path)
#     file_basename_lst = next(os.walk(os.path.join(data_root, "train")))[2]
#     train_file_path_lst = [os.path.join(data_root, "train", x) for x in file_basename_lst]
#     label_file_path_lst = [os.path.join(data_root, "label", x) for x in file_basename_lst]
#     images = []
#     labels = []
#     # write images
#     for i, file_path in enumerate(train_file_path_lst):
#         im = Image.open(file_path)
#         im = np.array(im, dtype=np.uint8)
#         if len(im.shape) == 3:
#             im = im[:, :, 0]
#         im = np.expand_dims(im, axis=2)
#         images.append(im)
#         if i % 10 == 0 and i > 0:
#             print("loading images... ", i)
#         if i % 10 == 0 and i > 0:
#             lmdb_x.write(images, None, None, "images")
#             images.clear()
#             print("images clear")
#     lmdb_x.write(images, None, None, "images")
#     # write labels
#     print("-" * 100)
#     for i, file_path in enumerate(label_file_path_lst):
#         im = Image.open(file_path)
#         im = np.array(im, dtype=np.uint8)
#         if len(im.shape) == 3:
#             im = im[:, :, 0]
#         im = np.expand_dims(im, axis=2)
#         labels.append(im)
#         if i % 10 == 0 and i > 0:
#             print("loading images... ", i)
#         if i % 10 == 0 and i > 0:
#             lmdb_y.write(labels, None, None, "images")
#             labels.clear()
#             print("images clear")
#     lmdb_y.write(labels, None, None, "images")
#     print(lmdb_x.count())
#     print(lmdb_y.count())


def make_hdf5_database():
    data_root = "/media/topsky/HHH/jzhang_root/data/njai/cbct"
    hdf5_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data.hdf5"
    f = h5py.File(hdf5_path, "w")
    grp_x = f.create_group("images")
    grp_y = f.create_group("labels")
    file_basename_lst = next(os.walk(os.path.join(data_root, "train")))[2]
    train_file_path_lst = [os.path.join(data_root, "train", x) for x in file_basename_lst]
    label_file_path_lst = [os.path.join(data_root, "label", x) for x in file_basename_lst]

    for i in range(len(file_basename_lst)):
        idx = '{:08}'.format(i + 1)
        image = Image.open(train_file_path_lst[i])
        image = np.array(image, dtype=np.uint8)
        if len(image.shape) == 3:
            image = image[:, :, 0]
        grp_x.create_dataset(idx, dtype=np.uint8, data=image)  # [0, 255]

        label = Image.open(label_file_path_lst[i])
        label = np.array(label, dtype=np.uint8)
        if len(label.shape) == 3:
            label = label[:, :, 0]
        grp_y.create_dataset(idx, dtype=np.uint8, data=label)
        if i % 10 == 0:
            print("loading images and labels... ", i)


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
    split = int(len(keys) * 0.8)
    for name in f:
        print(name)
    f.create_dataset("train_id", data=keys[:split])
    f.create_dataset("val_id", data=keys[split:])
    v = f["train_id"].value
    v = [x.tostring().decode() for x in v]
    print(v)
    print(len(v))
    f.close()


def generate_data_custom_test():
    hdf5_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data.hdf5"
    # gen = Data_Generator(hdf5_path, 8)
    f = DataGeneratorCustom(hdf5_path, 8, mode="train")
    g = f.gen.itt
    print(len(f))
    print(type(f))
    for i in range(2):
        # print(len(images) == len(labels))
        # print(len(images))
        images, labels = next(g)
        print(len(images))
        print(images[0].shape)
        # print(labels[0].shape)


def __main():
    # make_hdf5_database()
    # add_train_val_id_hdf5()
    # save_mask_image()
    # show_image()
    save_labelme2_mask()


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
