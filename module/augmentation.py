#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-7-31
# Task: 
# Insights: 

from datetime import datetime
from functools import wraps

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

from func.data_io import DataSet
from func.model_densenet import get_densenet121_unet_sigmoid_gn
from tqdm import tqdm

MASK_FILE_LIST = ['03+261mask.tif', '03+262mask.tif', '04+246mask.tif', '04+248mask.tif', '04+251mask.tif']

IMAGE_FILE_LIST = ['03+261ori.tif', '03+262ori.tif', '04+246ori.tif', '04+248ori.tif', '04+251ori.tif']


def erode_dilate_op():
    # img = cv2.imread("/home/topsky/Desktop/04+251mask.tif", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(
        "/media/topsky/HHH/jzhang_root/data/njai/cbct/CBCT_testingset/CBCT_testingset_pred20180801_42/04+246mask.tif",
        cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    print(h * w)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义结构元素
    # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
    result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算
    cv2.imwrite("/home/topsky/Desktop/test.tif", result)
    diff = (result - img) * 1
    print(sum(diff.flatten()))
    result = np.concatenate([img, result], axis=0)
    plt.imshow(result, "gray")
    plt.show()


def fill_contour(image_contour):
    """
    填充轮廓。
    :param image_contour: str or 2-d numpy array
    :return:
    """
    if isinstance(image_contour, str):
        img_in = cv2.imread(image_contour, cv2.IMREAD_GRAYSCALE)
    else:
        img_in = image_contour
    th, im_th = cv2.threshold(img_in, 127, 255, cv2.THRESH_BINARY)
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    return im_out


def get_pixel_wise_diff(image1, image2):
    assert image1.shape == image2.shape
    diff = (image1 != image2) * 1
    diff = sum(diff.flatten())
    return diff


def do_get_cop_acc():
    pred_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/CBCT_testingset/CBCT_testingset_pred20180801_42/best_val_loss_se_densenet_gn_fold01_1i_2o_20180730/mask0"
    mask_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/CBCT_testingset/CBCT_testingset_pred20180801_42"
    # mask_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/CBCT_testingset_custom_masks"
    pred_file_path_list = [os.path.join(pred_dir, x) for x in MASK_FILE_LIST]
    mask_file_path_list = [os.path.join(mask_dir, x) for x in IMAGE_FILE_LIST]
    diffs = 0
    total_pixels = 0
    for i in range(len(pred_file_path_list)):
        pred = cv2.imread(pred_file_path_list[i], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_file_path_list[i], cv2.IMREAD_GRAYSCALE)
        diffs += get_pixel_wise_diff(pred, mask)
        total_pixels += pred.size
    acc = (total_pixels - diffs) / total_pixels
    print("acc:", acc)


def do_fill_contour():
    contour = "/media/topsky/HHH/jzhang_root/data/njai/cbct/CBCT_testingset/CBCT_testingset_pred20180801_42/04+246mask_ori.tif"
    contour = cv2.imread(contour, cv2.IMREAD_GRAYSCALE)
    image = fill_contour(contour)
    print(get_pixel_wise_diff(contour, image))
    # plt.imshow(contour, "gray")
    # plt.show()


def do_convert_custom_label2contour():
    test_image_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/CBCT_testingset/CBCT_testingset"
    custom_label_image_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/CBCT_testingset_custom_label"
    contour_save_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/CBCT_testingset_custom_contour"
    if not os.path.isdir(contour_save_dir):
        os.makedirs(contour_save_dir)
    file_lst = next(os.walk(custom_label_image_dir))[2]
    contour_file_path_lst = [os.path.join(custom_label_image_dir, x) for x in file_lst]
    test_image_file_path_lst = [os.path.join(test_image_dir, x) for x in file_lst]
    image_contour_file_path_lst = [os.path.join(contour_save_dir, x) for x in file_lst]
    for i in range(len(contour_file_path_lst)):

        img_ori = cv2.imread(test_image_file_path_lst[i], cv2.IMREAD_UNCHANGED)
        img_contour = cv2.imread(contour_file_path_lst[i], cv2.IMREAD_UNCHANGED)[..., :3]

        is_same = img_ori == img_contour
        is_same = is_same * 1

        h, w, c = img_contour.shape
        img_contour = np.zeros(shape=(h, w))
        for j in range(h):
            for k in range(w):
                sum_ = sum(is_same[j, k])
                if sum_ != 3:
                    img_contour[j, k] = 255
        cv2.imwrite(image_contour_file_path_lst[i], img_contour)


def do_conver_contour2mask():
    contour_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/CBCT_testingset/CBCT_testingset_pred20180801_42/best_val_loss_se_densenet_gn_fold01_1i_2o_20180730/mask0"
    mask_save_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/CBCT_testingset/CBCT_testingset_pred20180801_42/best_val_loss_se_densenet_gn_fold01_1i_2o_20180730/mask0_fill_contour"
    if not os.path.isdir(mask_save_dir):
        os.makedirs(mask_save_dir)
    file_lst = next(os.walk(contour_dir))[2]
    contour_file_path_lst = [os.path.join(contour_dir, x) for x in file_lst]
    mask_file_path_lst = [os.path.join(mask_save_dir, x) for x in file_lst]
    for i in range(len(contour_file_path_lst)):
        mask = fill_contour(contour_file_path_lst[i])
        cv2.imwrite(mask_file_path_lst[i], mask)
    print("Done.")


def rotate(image, angle):
    if angle == 0 or -0:
        return image
    height, width = image.shape[0:2]
    mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    img = cv2.warpAffine(image, mat, (width, height),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)
    return img


def rotate_images(images, angle):
    rotated_images = np.zeros_like(images)
    for i in range(len(images)):
        rotated_image = rotate(images[i], angle)
        rotated_images[i] = rotated_image
    return rotated_images


def hflip(img, is_flip=1):
    assert len(img.shape) == 2 or len(img.shape) == 3
    if is_flip:
        return np.flip(img, 1)
    else:
        return img


def hflip_images(images, is_flip=1):
    assert len(images.shape) == 4
    if is_flip:
        return np.flip(images, 2)
    else:
        return images


def tta_test():
    img = cv2.imread("/media/topsky/HHH/jzhang_root/data/njai/cbct/CBCT_testingset/CBCT_testingset/04+246ori.tif",
                     cv2.IMREAD_GRAYSCALE)
    img = np.expand_dims(img, axis=-1)
    img = DataSet.preprocess(img, mode="image")
    img = np.expand_dims(img, axis=0)

    model = get_densenet121_unet_sigmoid_gn(input_shape=(None, None, 1), output_channels=2, weights=None)
    model.load_weights(
        "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/20180731_0/best_val_acc_se_densenet_gn_fold0_random_0_1i_2o_20180801.h5")
    pred = tta_predict(model, img, batch_size=1)
    # print(pred)
    pred = np.squeeze(pred, 0)
    print(pred.shape)
    pred = np.where(pred > 0.5, 255, 0)
    cv2.imwrite("/home/topsky/Desktop/mask_04+246ori_f1_random.tif", pred[..., 0])
    # plt.imshow(pred[..., 0], "gray")
    # plt.show()


def tta_predict(model, images, batch_size=1):
    """
    对预处理过的图片进行预测
    :param images: 4-d numpy array, pre-processed image. (b, h, w, c)
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
        pred = model.predict(imgs, batch_size=batch_size)
        pred = rotate_images(pred, -aug[1])
        pred = hflip_images(pred, aug[0])
        preds.append(pred)
    preds = np.array(preds)  # (nb_aug, b, h, w, c)

    preds_0 = preds[..., 0]  # (nb_aug, b, h, w)
    preds_1 = preds[..., 1]  # (nb_aug, b, h, w)

    pred_0 = np.zeros(shape=(bs, height, width), dtype=np.float64)
    pred_1 = np.zeros(shape=(bs, height, width), dtype=np.float64)
    for b in range(bs):
        pred_b_0 = preds_0[:, b, :, :]  # (nb_aug, h, w)
        pred_b_0 = np.mean(pred_b_0, axis=0)  # (h, w)
        pred_0[b] = pred_b_0

        pred_b_1 = preds_1[:, b, :, :]  # (nb_aug, h, w)
        pred_b_1 = np.mean(pred_b_1, axis=0)  # (h, w)
        pred_1[b] = pred_b_1

    pred_0 = np.expand_dims(pred_0, axis=-1)
    pred_1 = np.expand_dims(pred_1, axis=-1)
    pred = np.concatenate([pred_0, pred_1], axis=-1)
    return pred


# not use
def vflip(img):
    return np.flip(img, 0)


# not use
def random_flip(img, code):
    return cv2.flip(img, code)


# not use
def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


# not use
def clipped(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype, maxval = img.dtype, np.max(img)
        return clip(func(img, *args, **kwargs), dtype, maxval)

    return wrapped_function


# not use
@clipped
def random_brightness(img, alpha):
    return alpha * img


# not use
@clipped
def random_contrast(img, alpha):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    return alpha * img + gray


def __main():
    np.set_printoptions(threshold=np.inf)
    # erode_dilate_op()
    # do_fill_contour()
    # do_convert_custom_label2contour()
    do_conver_contour2mask()
    # do_get_cop_acc()
    # compare_image()
    # tta_test()
    pass


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
