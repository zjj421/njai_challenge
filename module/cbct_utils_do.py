#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-6-20
# Task: 
# Insights:
import glob
import json
import os
import subprocess
from datetime import datetime
from pprint import pprint

import cv2
import h5py
import numpy as np
import scipy.misc
from PIL import Image
from func.model_inception_resnet_v2_gn import get_inception_resnet_v2_unet_sigmoid_gn
from tqdm import tqdm

from module.inference import ModelDeployment
from module.utils_public import apply_mask, get_file_path_list, random_colors
import pandas as pd
import matplotlib.pyplot as plt


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
    data_dir = "/media/zj/share/data/njai_2018/cbct"
    images_dir = os.path.join(data_dir, "train")
    masks_dir = os.path.join(data_dir, "label")
    # 新合并图的保存路径。
    mask_image_dir = os.path.join(data_dir, "result", "image_mask_20180726")
    if not os.path.isdir(mask_image_dir):
        os.makedirs(mask_image_dir)
    basename_lst = next(os.walk(images_dir))[2]
    color = [0, 191, 255]
    alpha = 0.5
    for basename in basename_lst:
        image = Image.open(os.path.join(images_dir, basename))
        image = np.array(image)
        # image = cv2.imread(os.path.join(images_dir, basename))
        if len(image.shape) == 2:
            image = np.concatenate([np.expand_dims(image, axis=-1) for i in range(3)], axis=-1)
        mask = Image.open(os.path.join(masks_dir, basename))
        mask = np.array(mask)
        if len(mask.shape) == 3:
            assert mask[:, :, 0].all() == mask[:, :, 1].all() and mask[:, :, 0].all() == mask[:, :, 2].all()
            mask = mask[:, :, 0]
        mask = np.where(mask > 0.5, 255, 0)

        # mask = mask // 255
        mask_image = apply_mask(image, mask, color, alpha=alpha)
        # mask_image是否与image一起显示。
        mask_image = np.concatenate([image, mask_image], axis=1)
        scipy.misc.toimage(mask_image, cmin=0.0, cmax=...).save(os.path.join(mask_image_dir, basename))
        # cv2.imwrite(os.path.join(mask_image_dir, basename), mask_image)


def make_hdf5_database():
    """
    将image与mask保存在hdf5数据库里。
    Returns:

    """
    data_root = "/media/zj/share/data/njai_2018/cbct"
    hdf5_path = "/home/zj/helloworld/study/njai_challenge/cbct/inputs/data_0717.hdf5"
    f = h5py.File(hdf5_path, "w")
    grp_x = f.create_group("images")
    grp_y_0 = f.create_group("mask_0")
    grp_y_1 = f.create_group("mask_1")
    file_basename_lst = next(os.walk(os.path.join(data_root, "train")))[2]
    image_file_path_lst = [os.path.join(data_root, "train", x) for x in file_basename_lst]
    mask_0_file_path_lst = [os.path.join(data_root, "label", x) for x in file_basename_lst]
    mask_1_file_path_lst = [os.path.join(data_root, "gum_mask", x) for x in file_basename_lst]
    print("Saving images and labels ... ")
    # 更新:idx与实际名称对应起来。
    # 改用opencv读取图片。
    for i in tqdm(range(len(file_basename_lst))):
        idx = '{:08}'.format(int(os.path.splitext(file_basename_lst[i])[0]))
        # 保存image,原样保存。
        image = cv2.imread(image_file_path_lst[i], cv2.IMREAD_UNCHANGED)
        grp_x.create_dataset(idx, dtype=np.uint8, data=image)  # [0, 255]
        # 保存mask_tooth_root,只保存灰度图片. [0 or 255]
        mask_0 = cv2.imread(mask_0_file_path_lst[i], cv2.IMREAD_GRAYSCALE)
        grp_y_0.create_dataset(idx, dtype=np.uint8, data=mask_0)
        # 保存mask_gum,只保存灰度图片. [0 or 255]
        mask_1 = cv2.imread(mask_1_file_path_lst[i], cv2.IMREAD_GRAYSCALE)
        grp_y_1.create_dataset(idx, dtype=np.uint8, data=mask_1)


def add_k_fold_map_hdf5():
    hdf5_path = "/home/zj/helloworld/study/njai_challenge/cbct/inputs/data_0717.hdf5"
    f_h5 = h5py.File(hdf5_path, mode="r+")
    # del f_h5["k_fold_map"]
    k_fold_path = "/home/zj/helloworld/study/njai_challenge/module/k_fold.csv"
    with open(k_fold_path, 'r') as f:
        lines = f.readlines()
    k_fold_map = {}
    for line in lines:
        value, idx = line.split(",")
        value_start_end = value.strip().split("-")
        value_lst = ["{:08}".format(x) for x in range(int(value_start_end[0]), int(value_start_end[1]) + 1)]
        idx = idx.strip()
        if idx not in k_fold_map.keys():
            k_fold_map[idx] = value_lst
        else:
            k_fold_map[idx].extend(value_lst)
    json_obj = json.dumps(k_fold_map)
    f_h5.create_dataset("k_fold_map", data=json_obj)


# not use any more
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


def combine_image_mask_predict():
    image_mask_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/image_mask_20180714"
    pred_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/pred_mask_images_20180714"
    image_mask_pred_save_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/image_mask_pred_20180714"
    if not os.path.isdir(image_mask_pred_save_dir):
        os.makedirs(image_mask_pred_save_dir)
    pred_path_lst = get_file_path_list(pred_dir, ext='.tif')
    file_lst = [os.path.basename(x) for x in pred_path_lst]
    image_mask_path_lst = [os.path.join(image_mask_dir, x) for x in file_lst]
    for i in tqdm(range(len(pred_path_lst))):
        image_mask = np.array(Image.open(image_mask_path_lst[i]))
        pred = np.array(Image.open(pred_path_lst[i]))
        image_mask_pred = np.concatenate([image_mask, pred], axis=1)
        scipy.misc.toimage(image_mask_pred, cmin=0.0, cmax=...).save(
            os.path.join(image_mask_pred_save_dir, file_lst[i]))


def read_data_and_show():
    data_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data.hdf5"
    val_id_save_path = "/media/topsky/HHH/jzhang_root/data/njai/cbct/pred_mask_images_20180711/val_id.csv"
    f = h5py.File(data_path, mode="r")
    v = f["val_id"].value
    v = [x.tostring().decode() for x in v]
    v = sorted(v)
    print(v)
    mask1_group = f["mask_1"]
    mask = mask1_group["00000001"]
    print(mask.shape)
    print(np.array(mask))
    plt.figure()
    plt.imshow(mask[..., 0], cmap="gray")
    plt.show()


def map_file2index():
    data_root = "/media/topsky/HHH/jzhang_root/data/njai/cbct"
    file_basename_lst = next(os.walk(os.path.join(data_root, "train")))[2]
    idx2file = {}
    for i in range(len(file_basename_lst)):
        idx = '{:08}'.format(i + 1)
        idx2file[idx] = file_basename_lst[i]
    pprint(idx2file)


def predict_and_save_stage1_result():
    h5_data_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data_0717.hdf5"
    stage1_h5_data = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/predicted_masks_data.hdf5"
    model_weights = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/final_inception_resnet_v2_gn_fold1_1i_1o_stage1.h5"
    model_def = get_inception_resnet_v2_unet_sigmoid_gn(weights=None, output_channels=1)
    model_obj = ModelDeployment(model_def, model_weights)
    model_obj.predict_and_save_stage1_masks(h5_data_path, stage1_h5_data, fold_k=1, batch_size=4)
    # model_obj.predict_from_h5data(h5_data_path, val_fold_nb=1, is_train=False, save_dir=result_save_dir)
    # model_obj.predict_from_h5data(h5_data_path, val_fold_nb=1, is_train=True, save_dir=result_save_dir)

def __main():
    np.set_printoptions(threshold=np.inf)
    # make_hdf5_database()
    # add_train_val_id_hdf5()
    # save_mask_image()
    # show_image()
    # save_labelme2_mask()
    # read_data_test()
    # combine_image_mask_predict()
    # read_data_and_show()
    # map_file2index()
    # add_k_fold_map_hdf5()
    predict_and_save_stage1_result()
    pass


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
